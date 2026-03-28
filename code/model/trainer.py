from __future__ import absolute_import
from __future__ import division
from tqdm import tqdm
import json
import time
import os
import logging
import numpy as np
import torch
import codecs
from collections import defaultdict
import gc
import resource
import sys
from scipy.special import logsumexp as lse

from code.model.agent import Agent
from code.options import read_options
from code.model.environment import env
from code.model.baseline import ReactiveBaseline
from code.model.nell_eval import nell_eval

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Trainer(object):
    def __init__(self, params):
        # transfer parameters to self
        for key, val in params.items():
            setattr(self, key, val)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent = Agent(params).to(self.device)
        self.save_path = None
        self.train_environment = env(params, "train")
        self.dev_test_environment = env(params, "dev")
        self.test_test_environment = env(params, "test")
        self.test_environment = self.dev_test_environment
        self.rev_relation_vocab = self.train_environment.grapher.rev_relation_vocab
        self.rev_entity_vocab = self.train_environment.grapher.rev_entity_vocab
        self.max_hits_at_10 = 0
        self.ePAD = self.entity_vocab["PAD"]
        self.rPAD = self.relation_vocab["PAD"]

        # optimize
        self.baseline = ReactiveBaseline(l=self.Lambda)
        trainable_params = [p for p in self.agent.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)

    def initialize(self, restore=None):
        if restore:
            self._load_checkpoint(restore)

    def _resolve_checkpoint_path(self, path):
        if os.path.isdir(path):
            candidate = os.path.join(path, "model.ckpt.pt")
            if os.path.exists(candidate):
                return candidate
            raise ValueError("No PyTorch checkpoint found at {}".format(candidate))

        if os.path.exists(path):
            return path

        with_suffix = path + ".pt"
        if os.path.exists(with_suffix):
            return with_suffix

        raise ValueError(
            "Checkpoint not found: {}. TensorFlow .ckpt files are not loadable in the PyTorch port.".format(path)
        )

    def _save_checkpoint(self):
        save_path = os.path.join(self.model_dir, "model.ckpt.pt")
        torch.save(
            {
                "model_state_dict": self.agent.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "max_hits_at_10": self.max_hits_at_10,
            },
            save_path,
        )
        return save_path

    def _load_checkpoint(self, restore):
        ckpt_path = self._resolve_checkpoint_path(restore)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        if "model_state_dict" not in checkpoint:
            raise ValueError(
                "Invalid checkpoint format at {}. Expected a PyTorch checkpoint created by this trainer.".format(
                    ckpt_path
                )
            )
        self.agent.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.max_hits_at_10 = checkpoint.get("max_hits_at_10", self.max_hits_at_10)

    def initialize_pretrained_embeddings(self):
        if self.pretrained_embeddings_action != "":
            embeddings = np.loadtxt(open(self.pretrained_embeddings_action))
            embeddings = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                self.agent.relation_lookup_table.weight.copy_(embeddings)
        if self.pretrained_embeddings_entity != "":
            embeddings = np.loadtxt(open(self.pretrained_embeddings_entity))
            embeddings = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                self.agent.entity_lookup_table.weight.copy_(embeddings)

    def entropy_reg_loss(self, all_logits):
        all_logits = torch.stack(all_logits, dim=2)  # [B, MAX_NUM_ACTIONS, T]
        entropy_policy = -torch.mean(torch.sum(torch.exp(all_logits) * all_logits, dim=1))
        return entropy_policy

    def calc_reinforce_loss(self, per_example_loss, per_example_logits, cum_discounted_reward):
        loss = torch.stack(per_example_loss, dim=1)  # [B, T]

        baseline_val = self.baseline.get_baseline_value()
        final_reward = cum_discounted_reward - baseline_val

        reward_mean = torch.mean(final_reward)
        reward_var = torch.var(final_reward, unbiased=False)
        reward_std = torch.sqrt(reward_var) + 1e-6
        final_reward = (final_reward - reward_mean) / reward_std

        loss = loss * final_reward

        decaying_beta = self.beta * (0.90 ** (self.batch_counter / 200.0))
        total_loss = torch.mean(loss) - decaying_beta * self.entropy_reg_loss(per_example_logits)
        return total_loss

    def calc_cum_discounted_reward(self, rewards):
        running_add = np.zeros([rewards.shape[0]])
        cum_disc_reward = np.zeros([rewards.shape[0], self.path_length])
        cum_disc_reward[:, self.path_length - 1] = rewards
        for t in reversed(range(self.path_length)):
            running_add = self.gamma * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add
        return cum_disc_reward

    def train(self):
        self.agent.train()

        train_loss = 0.0
        self.batch_counter = 0

        for episode in self.train_environment.get_episodes():
            self.batch_counter += 1

            state = episode.get_state()
            batch_total = state["current_entities"].shape[0]
            range_arr = torch.arange(batch_total, dtype=torch.long, device=self.device)

            query_relation = torch.tensor(episode.get_query_relation(), dtype=torch.long, device=self.device)
            prev_state = self.agent.init_memory(batch_total, self.device)
            prev_relation = self.agent.get_dummy_start_label(batch_total, self.device)

            per_example_loss = []
            per_example_logits = []

            for _ in range(self.path_length):
                next_relations = torch.tensor(state["next_relations"], dtype=torch.long, device=self.device)
                next_entities = torch.tensor(state["next_entities"], dtype=torch.long, device=self.device)
                current_entities = torch.tensor(state["current_entities"], dtype=torch.long, device=self.device)

                loss_t, prev_state, logits_t, idx_t, _ = self.agent.step(
                    next_relations,
                    next_entities,
                    prev_state,
                    prev_relation,
                    self.agent.relation_lookup_table(query_relation),
                    current_entities,
                    label_action=None,
                    range_arr=range_arr,
                    first_step_of_test=False,
                )

                per_example_loss.append(loss_t)
                per_example_logits.append(logits_t)
                prev_relation = idx_t.new_tensor(next_relations[range_arr, idx_t])

                state = episode(idx_t.detach().cpu().numpy())

            rewards = episode.get_reward()
            cum_discounted_reward = self.calc_cum_discounted_reward(rewards)
            cum_discounted_reward = torch.tensor(cum_discounted_reward, dtype=torch.float32, device=self.device)

            total_loss = self.calc_reinforce_loss(per_example_loss, per_example_logits, cum_discounted_reward)

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            self.baseline.update(torch.mean(cum_discounted_reward))

            batch_total_loss = float(total_loss.detach().item())
            train_loss = 0.98 * train_loss + 0.02 * batch_total_loss
            avg_reward = np.mean(rewards)

            reward_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))
            reward_reshape = np.sum(reward_reshape, axis=1)
            reward_reshape = reward_reshape > 0
            num_ep_correct = np.sum(reward_reshape)

            if np.isnan(train_loss):
                raise ArithmeticError("Error in computing loss")

            logger.info(
                "batch_counter: {0:4d}, num_hits: {1:7.4f}, avg. reward per batch {2:7.4f}, "
                "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, train loss {5:7.4f}".format(
                    self.batch_counter,
                    np.sum(rewards),
                    avg_reward,
                    num_ep_correct,
                    (num_ep_correct / self.batch_size),
                    train_loss,
                )
            )

            if self.batch_counter % self.eval_every == 0:
                with open(self.output_dir + "/scores.txt", "a") as score_file:
                    score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
                os.mkdir(self.path_logger_file + "/" + str(self.batch_counter))
                self.path_logger_file_ = self.path_logger_file + "/" + str(self.batch_counter) + "/paths"

                self.test(beam=True, print_paths=False)

            logger.info("Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            gc.collect()
            if self.batch_counter >= self.total_iterations:
                break

    def test(self, beam=False, print_paths=False, save_model=True, auc=False):
        self.agent.eval()

        batch_counter = 0
        paths = defaultdict(list)
        answers = []
        all_final_reward_1 = 0
        all_final_reward_3 = 0
        all_final_reward_5 = 0
        all_final_reward_10 = 0
        all_final_reward_20 = 0
        auc = 0

        total_examples = self.test_environment.total_no_examples

        with torch.no_grad():
            for episode in tqdm(self.test_environment.get_episodes()):
                batch_counter += 1

                temp_batch_size = episode.no_examples
                self.qr = episode.get_query_relation()
                state = episode.get_state()

                beam_probs = np.zeros((temp_batch_size * self.test_rollouts, 1))

                batch_total = temp_batch_size * self.test_rollouts
                query_relation = torch.tensor(self.qr, dtype=torch.long, device=self.device)
                range_arr = torch.arange(batch_total, dtype=torch.long, device=self.device)
                query_embedding = self.agent.relation_lookup_table(query_relation)

                agent_mem = self.agent.init_memory(batch_total, self.device)
                previous_relation = self.agent.get_dummy_start_label(batch_total, self.device)

                if print_paths:
                    self.entity_trajectory = []
                    self.relation_trajectory = []

                self.log_probs = np.zeros((batch_total,)) * 1.0

                for i in range(self.path_length):
                    next_relations_t = torch.tensor(state["next_relations"], dtype=torch.long, device=self.device)
                    next_entities_t = torch.tensor(state["next_entities"], dtype=torch.long, device=self.device)
                    current_entities_t = torch.tensor(state["current_entities"], dtype=torch.long, device=self.device)

                    _, agent_mem, test_scores_t, test_action_idx_t, chosen_relation_t = self.agent.step(
                        next_relations_t,
                        next_entities_t,
                        agent_mem,
                        previous_relation,
                        query_embedding,
                        current_entities_t,
                        label_action=None,
                        range_arr=range_arr,
                        first_step_of_test=(i == 0),
                    )

                    test_scores = test_scores_t.detach().cpu().numpy()
                    test_action_idx = test_action_idx_t.detach().cpu().numpy()
                    chosen_relation = chosen_relation_t.detach().cpu().numpy()

                    if beam:
                        k = self.test_rollouts
                        new_scores = test_scores + beam_probs
                        if i == 0:
                            idx = np.argsort(new_scores)
                            idx = idx[:, -k:]
                            ranged_idx = np.tile([b for b in range(k)], temp_batch_size)
                            idx = idx[np.arange(k * temp_batch_size), ranged_idx]
                        else:
                            idx = self.top_k(new_scores, k)

                        y = idx // self.max_num_actions
                        x = idx % self.max_num_actions

                        y += np.repeat([b * k for b in range(temp_batch_size)], k)
                        state["current_entities"] = state["current_entities"][y]
                        state["next_relations"] = state["next_relations"][y, :]
                        state["next_entities"] = state["next_entities"][y, :]

                        y_t = torch.tensor(y, dtype=torch.long, device=self.device)
                        agent_mem = (agent_mem[0][:, y_t, :], agent_mem[1][:, y_t, :])

                        test_action_idx = x
                        chosen_relation = state["next_relations"][np.arange(temp_batch_size * k), x]
                        beam_probs = new_scores[y, x]
                        beam_probs = beam_probs.reshape((-1, 1))
                        if print_paths:
                            for j in range(i):
                                self.entity_trajectory[j] = self.entity_trajectory[j][y]
                                self.relation_trajectory[j] = self.relation_trajectory[j][y]

                    previous_relation = torch.tensor(chosen_relation, dtype=torch.long, device=self.device)

                    if print_paths:
                        self.entity_trajectory.append(state["current_entities"])
                        self.relation_trajectory.append(chosen_relation)

                    state = episode(test_action_idx)
                    self.log_probs += test_scores[np.arange(self.log_probs.shape[0]), test_action_idx]

                if beam:
                    self.log_probs = beam_probs

                if print_paths:
                    self.entity_trajectory.append(state["current_entities"])

                rewards = episode.get_reward()
                reward_reshape = np.reshape(rewards, (temp_batch_size, self.test_rollouts))
                self.log_probs = np.reshape(self.log_probs, (temp_batch_size, self.test_rollouts))
                sorted_indx = np.argsort(-self.log_probs)
                final_reward_1 = 0
                final_reward_3 = 0
                final_reward_5 = 0
                final_reward_10 = 0
                final_reward_20 = 0
                AP = 0
                ce = episode.state["current_entities"].reshape((temp_batch_size, self.test_rollouts))
                se = episode.start_entities.reshape((temp_batch_size, self.test_rollouts))

                for b in range(temp_batch_size):
                    answer_pos = None
                    seen = set()
                    pos = 0
                    if self.pool == "max":
                        for r in sorted_indx[b]:
                            if reward_reshape[b, r] == self.positive_reward:
                                answer_pos = pos
                                break
                            if ce[b, r] not in seen:
                                seen.add(ce[b, r])
                                pos += 1

                    if self.pool == "sum":
                        scores = defaultdict(list)
                        answer = ""
                        for r in sorted_indx[b]:
                            scores[ce[b, r]].append(self.log_probs[b, r])
                            if reward_reshape[b, r] == self.positive_reward:
                                answer = ce[b, r]
                        final_scores = defaultdict(float)
                        for e in scores:
                            final_scores[e] = lse(scores[e])
                        sorted_answers = sorted(final_scores, key=final_scores.get, reverse=True)
                        if answer in sorted_answers:
                            answer_pos = sorted_answers.index(answer)
                        else:
                            answer_pos = None

                    if answer_pos is not None:
                        if answer_pos < 20:
                            final_reward_20 += 1
                            if answer_pos < 10:
                                final_reward_10 += 1
                                if answer_pos < 5:
                                    final_reward_5 += 1
                                    if answer_pos < 3:
                                        final_reward_3 += 1
                                        if answer_pos < 1:
                                            final_reward_1 += 1
                    if answer_pos is None:
                        AP += 0
                    else:
                        AP += 1.0 / (answer_pos + 1)

                    if print_paths:
                        qr = self.train_environment.grapher.rev_relation_vocab[self.qr[b * self.test_rollouts]]
                        start_e = self.rev_entity_vocab[episode.start_entities[b * self.test_rollouts]]
                        end_e = self.rev_entity_vocab[episode.end_entities[b * self.test_rollouts]]
                        paths[str(qr)].append(str(start_e) + "\t" + str(end_e) + "\n")
                        paths[str(qr)].append(
                            "Reward:" + str(1 if answer_pos is not None and answer_pos < 10 else 0) + "\n"
                        )
                        for r in sorted_indx[b]:
                            indx = b * self.test_rollouts + r
                            if rewards[indx] == self.positive_reward:
                                rev = 1
                            else:
                                rev = -1
                            answers.append(
                                self.rev_entity_vocab[se[b, r]]
                                + "\t"
                                + self.rev_entity_vocab[ce[b, r]]
                                + "\t"
                                + str(self.log_probs[b, r])
                                + "\n"
                            )
                            paths[str(qr)].append(
                                "\t".join([str(self.rev_entity_vocab[e[indx]]) for e in self.entity_trajectory])
                                + "\n"
                                + "\t".join([str(self.rev_relation_vocab[re[indx]]) for re in self.relation_trajectory])
                                + "\n"
                                + str(rev)
                                + "\n"
                                + str(self.log_probs[b, r])
                                + "\n___\n"
                            )
                        paths[str(qr)].append("#####################\n")

                all_final_reward_1 += final_reward_1
                all_final_reward_3 += final_reward_3
                all_final_reward_5 += final_reward_5
                all_final_reward_10 += final_reward_10
                all_final_reward_20 += final_reward_20
                auc += AP

        all_final_reward_1 /= total_examples
        all_final_reward_3 /= total_examples
        all_final_reward_5 /= total_examples
        all_final_reward_10 /= total_examples
        all_final_reward_20 /= total_examples
        auc /= total_examples

        if save_model:
            if all_final_reward_10 >= self.max_hits_at_10:
                self.max_hits_at_10 = all_final_reward_10
                self.save_path = self._save_checkpoint()

        if print_paths:
            logger.info("[ printing paths at {} ]".format(self.output_dir + "/test_beam/"))
            for q in paths:
                j = q.replace("/", "-")
                with codecs.open(self.path_logger_file_ + "_" + j, "a", "utf-8") as pos_file:
                    for p in paths[q]:
                        pos_file.write(p)
            with open(self.path_logger_file_ + "answers", "w") as answer_file:
                for a in answers:
                    answer_file.write(a)

        with open(self.output_dir + "/scores.txt", "a") as score_file:
            score_file.write("Hits@1: {0:7.4f}".format(all_final_reward_1))
            score_file.write("\n")
            score_file.write("Hits@3: {0:7.4f}".format(all_final_reward_3))
            score_file.write("\n")
            score_file.write("Hits@5: {0:7.4f}".format(all_final_reward_5))
            score_file.write("\n")
            score_file.write("Hits@10: {0:7.4f}".format(all_final_reward_10))
            score_file.write("\n")
            score_file.write("Hits@20: {0:7.4f}".format(all_final_reward_20))
            score_file.write("\n")
            score_file.write("auc: {0:7.4f}".format(auc))
            score_file.write("\n")
            score_file.write("\n")

        logger.info("Hits@1: {0:7.4f}".format(all_final_reward_1))
        logger.info("Hits@3: {0:7.4f}".format(all_final_reward_3))
        logger.info("Hits@5: {0:7.4f}".format(all_final_reward_5))
        logger.info("Hits@10: {0:7.4f}".format(all_final_reward_10))
        logger.info("Hits@20: {0:7.4f}".format(all_final_reward_20))
        logger.info("auc: {0:7.4f}".format(auc))

    def top_k(self, scores, k):
        scores = scores.reshape(-1, k * self.max_num_actions)
        idx = np.argsort(scores, axis=1)
        idx = idx[:, -k:]
        return idx.reshape((-1))


if __name__ == "__main__":
    options = read_options()

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s: [ %(message)s ]", "%m/%d/%Y %I:%M:%S %p")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(options["log_file_name"], "w")
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    logger.info("reading vocab files...")
    options["relation_vocab"] = json.load(open(options["vocab_dir"] + "/relation_vocab.json"))
    options["entity_vocab"] = json.load(open(options["vocab_dir"] + "/entity_vocab.json"))
    logger.info("Reading mid to name map")
    logger.info("Done..")
    logger.info("Total number of entities {}".format(len(options["entity_vocab"])))
    logger.info("Total number of relations {}".format(len(options["relation_vocab"])))

    save_path = ""

    if not options["load_model"]:
        trainer = Trainer(options)
        trainer.initialize()
        trainer.initialize_pretrained_embeddings()

        trainer.train()
        save_path = trainer.save_path
        path_logger_file = trainer.path_logger_file
        output_dir = trainer.output_dir
    else:
        logger.info("Skipping training")
        logger.info("Loading model from {}".format(options["model_load_dir"]))

    trainer = Trainer(options)
    if options["load_model"]:
        save_path = options["model_load_dir"]
        path_logger_file = trainer.path_logger_file
        output_dir = trainer.output_dir

    trainer.initialize(restore=save_path)

    trainer.test_rollouts = 100

    os.mkdir(path_logger_file + "/" + "test_beam")
    trainer.path_logger_file_ = path_logger_file + "/" + "test_beam" + "/paths"
    with open(output_dir + "/scores.txt", "a") as score_file:
        score_file.write("Test (beam) scores with best model from " + str(save_path) + "\n")
    trainer.test_environment = trainer.test_test_environment
    trainer.test_environment.test_rollouts = 100

    trainer.test(beam=True, print_paths=True, save_model=False)

    print(options["nell_evaluation"])
    if options["nell_evaluation"] == 1:
        nell_eval(path_logger_file + "/" + "test_beam/" + "pathsanswers", trainer.data_input_dir + "/sort_test.pairs")
