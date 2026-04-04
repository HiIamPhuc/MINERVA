from __future__ import absolute_import
from __future__ import division
from tqdm import tqdm
import json
import time
import os
import numpy as np
import torch
import codecs
from collections import defaultdict
import gc
import resource
from scipy.special import logsumexp as lse
import sys

from agent import Agent
from options import read_options
from environment import Environment
from baseline import ReactiveBaseline
from nell_eval import nell_eval
try:
    from ..gwm.model import GWM
except ImportError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from model.gwm.model import GWM


class Trainer(object):
    def __init__(self, params):
        # transfer parameters to self
        for key, val in params.items():
            setattr(self, key, val)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = None

        self.agent = Agent(params).to(self.device)

        self.train_environment = Environment(params, "train")
        self.dev_test_environment = Environment(params, "dev")
        self.test_test_environment = Environment(params, "test")
        self.test_environment = self.dev_test_environment

        self.id2relation = self.train_environment.grapher.id2relation
        self.id2entity = self.train_environment.grapher.id2entity
        self.scores_file = os.path.join(self.output_dir, "scores.txt")

        self.max_hits_at_10 = 0
        
        self.ePAD = self.entity2id["PAD"]
        self.rPAD = self.relation2id["PAD"]

        # optimize
        self.baseline = ReactiveBaseline(l=self.baseline_decay)
        trainable_params = [p for p in self.agent.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)

    def _save_checkpoint(self):
        save_path = os.path.join(self.model_dir, "model.pt")
        torch.save(
            {
                "model_state_dict": self.agent.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "max_hits_at_10": self.max_hits_at_10,
            },
            save_path,
        )
        return save_path

    def entropy_reg_loss(self, all_logits):
        all_logits = torch.stack(all_logits, dim=2)  # [B, MAX_NUM_ACTIONS, T]
        entropy_policy = -torch.mean(torch.sum(torch.exp(all_logits) * all_logits, dim=1))
        return entropy_policy

    def calc_reinforce_loss(self, per_example_loss, per_example_logits, cum_discounted_reward):
        loss = torch.stack(per_example_loss, dim=1)  # [B, T]
        final_reward = cum_discounted_reward - self.baseline.get_baseline_value()

        reward_std, reward_mean = torch.std_mean(final_reward, unbiased=False)
        loss = loss * (final_reward - reward_mean) / (reward_std + 1e-6)

        decaying_entropy_weight = self.entropy_weight * (0.90 ** (self.batch_counter / 200.0))
        total_loss = torch.mean(loss) - decaying_entropy_weight * self.entropy_reg_loss(per_example_logits)
        return total_loss

    def calc_cum_discounted_reward(self, rewards):
        running_add = np.zeros([rewards.shape[0]])
        cum_disc_reward = np.zeros([rewards.shape[0], self.path_length])
        cum_disc_reward[:, self.path_length - 1] = rewards
        for t in reversed(range(self.path_length)):
            running_add = self.discount_factor * running_add + cum_disc_reward[:, t]
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
                    self.agent.lookup_relation(query_relation),
                    current_entities,
                    range_arr=range_arr,
                )

                per_example_loss.append(loss_t)
                per_example_logits.append(logits_t)
                prev_relation = next_relations[range_arr, idx_t].long()

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

            num_ep_correct = np.sum(rewards.reshape((self.batch_size, self.num_rollouts)).any(axis=1))

            if np.isnan(train_loss):
                raise ArithmeticError("Error in computing loss")

            print(
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

            if self.batch_counter % self.eval_interval == 0:
                with open(self.scores_file, "a", encoding="utf-8") as score_file:
                    score_file.write("Score for step " + str(self.batch_counter) + "\n\n")

                os.makedirs(self.path_logger_file + "/" + str(self.batch_counter), exist_ok=True)
                self.path_logger_file_ = self.path_logger_file + "/" + str(self.batch_counter) + "/paths"

                self.test(beam=True, print_paths=False)

            print("Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            gc.collect()
            if self.batch_counter >= self.total_steps:
                break

    def test(self, beam=False, print_paths=False, save_model=True):
        self.agent.eval()

        batch_counter = 0
        paths = defaultdict(list)
        answers = []
        metrics = {"Hits@1": 0.0, "Hits@3": 0.0, "Hits@5": 0.0, "Hits@10": 0.0, "Hits@20": 0.0, "mrr": 0.0, "auc": 0.0}

        total_examples = self.test_environment.total_no_examples

        with torch.no_grad():
            for episode in tqdm(self.test_environment.get_episodes()):
                batch_counter += 1

                temp_batch_size = episode.no_examples
                self.query_relation = episode.get_query_relation()
                state = episode.get_state()

                beam_probs = np.zeros((temp_batch_size * self.test_rollouts, 1))

                batch_total = temp_batch_size * self.test_rollouts
                query_relation_tensor = torch.tensor(self.query_relation, dtype=torch.long, device=self.device)
                range_arr = torch.arange(batch_total, dtype=torch.long, device=self.device)
                query_embedding = self.agent.lookup_relation(query_relation_tensor)

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
                        range_arr=range_arr,
                    )

                    test_scores = test_scores_t.detach().cpu().numpy()
                    test_action_idx = test_action_idx_t.detach().cpu().numpy()
                    chosen_relation = chosen_relation_t.detach().cpu().numpy()

                    if beam:
                        k = self.test_rollouts
                        new_scores = test_scores + beam_probs
                        action_dim = new_scores.shape[1]
                        if i == 0:
                            idx = np.argsort(new_scores)
                            idx = idx[:, -k:]
                            ranged_idx = np.tile([b for b in range(k)], temp_batch_size)
                            idx = idx[np.arange(k * temp_batch_size), ranged_idx]
                        else:
                            flattened_scores = torch.tensor(new_scores, device=self.device).view(-1, k * action_dim)
                            _, topk_idx = torch.topk(flattened_scores, k, dim=1, sorted=True)
                            idx = topk_idx.cpu().numpy().reshape(-1)

                        y = idx // action_dim
                        x = idx % action_dim

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
                
                ce = episode.state["current_entities"].reshape((temp_batch_size, self.test_rollouts))
                se = episode.start_entities.reshape((temp_batch_size, self.test_rollouts))

                for b in range(temp_batch_size):
                    answer_pos = None
                    seen = set()
                    pos = 0
                    if self.eval_pool_mode == "max":
                        for r in sorted_indx[b]:
                            if reward_reshape[b, r] == self.positive_reward:
                                answer_pos = pos
                                break
                            if ce[b, r] not in seen:
                                seen.add(ce[b, r])
                                pos += 1

                    if self.eval_pool_mode == "sum":
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
                        metrics["Hits@20"] += (answer_pos < 20)
                        metrics["Hits@10"] += (answer_pos < 10)
                        metrics["Hits@5"] += (answer_pos < 5)
                        metrics["Hits@3"] += (answer_pos < 3)
                        metrics["Hits@1"] += (answer_pos < 1)
                        metrics["auc"] += 1.0 / (answer_pos + 1)
                        metrics["mrr"] += 1.0 / (answer_pos + 1)

                    if print_paths:
                        query_relation_val = self.train_environment.grapher.id2relation[self.query_relation[b * self.test_rollouts]]
                        start_e = self.id2entity[episode.start_entities[b * self.test_rollouts]]
                        end_e = self.id2entity[episode.end_entities[b * self.test_rollouts]]
                        paths[str(query_relation_val)].append(str(start_e) + "\t" + str(end_e) + "\n")
                        paths[str(query_relation_val)].append(
                            "Reward:" + str(1 if answer_pos is not None and answer_pos < 10 else 0) + "\n"
                        )
                        for r in sorted_indx[b]:
                            indx = b * self.test_rollouts + r
                            if rewards[indx] == self.positive_reward:
                                rev = 1
                            else:
                                rev = -1
                            answers.append(
                                self.id2entity[se[b, r]]
                                + "\t"
                                + self.id2entity[ce[b, r]]
                                + "\t"
                                + str(self.log_probs[b, r])
                                + "\n"
                            )
                            paths[str(query_relation_val)].append(
                                "\t".join([str(self.id2entity[e[indx]]) for e in self.entity_trajectory])
                                + "\n"
                                + "\t".join([str(self.id2relation[re[indx]]) for re in self.relation_trajectory])
                                + "\n"
                                + str(rev)
                                + "\n"
                                + str(self.log_probs[b, r])
                                + "\n___\n"
                            )
                        paths[str(query_relation_val)].append("#####################\n")

        for metric_name in metrics:
            metrics[metric_name] /= total_examples

        if save_model:
            if metrics["Hits@10"] >= self.max_hits_at_10:
                self.max_hits_at_10 = metrics["Hits@10"]
                self.save_path = self._save_checkpoint()

        if print_paths:
            print("[ printing paths at {} ]".format(self.output_dir + "/test_beam/"))
            for q in paths:
                j = q.replace("/", "-")
                with codecs.open(self.path_logger_file_ + "_" + j, "a", "utf-8") as pos_file:
                    for p in paths[q]:
                        pos_file.write(p)
            with open(self.path_logger_file_ + "answers", "w") as answer_file:
                for a in answers:
                    answer_file.write(a)

        with open(self.scores_file, "a", encoding="utf-8") as score_file:
            for key, value in metrics.items():
                metric_line = "{0}: {1:7.4f}".format(key, value)
                print(metric_line)
                score_file.write(metric_line + "\n")
            score_file.write("\n")

    # Cleaned up redundant top_k in favor of torch.topk natively
    # def top_k(self, scores, k):
    #     scores = scores.reshape(-1, k * self.max_actions)
    #     idx = np.argsort(scores, axis=1)
    #     idx = idx[:, -k:]
    #     return idx.reshape((-1))


if __name__ == "__main__":
    options = read_options()

    print("Loaded config:", options.get("config_path", "<inline>"))
    print("Loading vocab from:", options["data_dir"])
    options["relation2id"] = json.load(open(options["data_dir"] + "/relation2id.json"))
    options["entity2id"] = json.load(open(options["data_dir"] + "/entity2id.json"))
    print(
        "Vocab loaded | entities={} relations={}".format(
            len(options["entity2id"]),
            len(options["relation2id"]),
        )
    )

    try:
        print("Loading pre-trained Graph World Model (GWM-RNN)...")
        gwm_weights_path = options.get("gwm_model_path", "")
        if not gwm_weights_path or not os.path.exists(gwm_weights_path):
            raise FileNotFoundError(
                f"GWM checkpoint not found. Please provide --gwm_model_path. Got: {gwm_weights_path}"
            )

        gwm_model = GWM.load_from_checkpoint(gwm_weights_path)
        gwm_model.eval()

        entity_emb_path = os.path.join(options["data_dir"], "entity_text_embeddings.pt")
        relation_emb_path = os.path.join(options["data_dir"], "relation_text_embeddings.pt")
        if not os.path.exists(entity_emb_path) or not os.path.exists(relation_emb_path):
            raise FileNotFoundError(
                "Missing entity_text_embeddings.pt or relation_text_embeddings.pt in data_dir. "
                "Run preprocess_data.py before training."
            )

        cache_device = options.get("gwm_text_cache_device", "cpu")
        gwm_model.load_precomputed_text_embedding_cache(
            entity_source=entity_emb_path,
            relation_source=relation_emb_path,
            cache_device=cache_device,
        )

        options["gwm_model"] = gwm_model
        options["hallucinate_k"] = options.get("hallucinate_k", 3)
    except ImportError as e:
        raise ImportError(f"Failed to import local GWM module: {e}")

    save_path = ""

    trainer = Trainer(options)

    trainer.train()
    save_path = trainer.save_path
    if not save_path:
        save_path = trainer._save_checkpoint()
    path_logger_file = trainer.path_logger_file
    output_dir = trainer.output_dir

    test_trainer = Trainer(options)
    test_trainer.agent.load_state_dict(torch.load(save_path, map_location=test_trainer.device)["model_state_dict"])
    
    test_trainer.test_rollouts = 100

    os.makedirs(path_logger_file + "/" + "test_beam", exist_ok=True)
    test_trainer.path_logger_file_ = path_logger_file + "/" + "test_beam" + "/paths"
    with open(test_trainer.scores_file, "a", encoding="utf-8") as score_file:
        score_file.write("Test (beam) scores with best model from " + str(save_path) + "\n")
    test_trainer.test_environment = test_trainer.test_test_environment
    test_trainer.test_environment.test_rollouts = 100

    test_trainer.test(beam=True, print_paths=True, save_model=False)

    print("NELL evaluation flag:", options["nell_evaluation"])
    if options["nell_evaluation"] == 1:
        nell_eval(path_logger_file + "/" + "test_beam/" + "pathsanswers", test_trainer.data_dir + "/sort_test.pairs")
