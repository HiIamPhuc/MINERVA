import os

class RunLogger:
    def __init__(self, scores_file, output_dir):
        self.scores_file = scores_file
        self.output_dir = output_dir

    @staticmethod
    def _append_lines(file_path, lines):
        with open(file_path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(str(line) + "\n")

    @staticmethod
    def _safe_lookup(mapping, item_id):
        item_id = int(item_id)
        return str(mapping.get(item_id, str(item_id)))

    def append_lines(self, lines):
        self._append_lines(self.scores_file, lines)

    def write_score_header(self, step):
        self._append_lines(self.scores_file, ["Score for step {}".format(step), ""])

    def write_metrics(self, metrics):
        metric_lines = []
        for key, value in metrics.items():
            metric_line = "{0}: {1:7.4f}".format(key, value)
            print(metric_line)
            metric_lines.append(metric_line)
        metric_lines.append("")
        self._append_lines(self.scores_file, metric_lines)

    def log_train_step(
        self,
        step,
        total_steps,
        progress_pct,
        remaining_steps,
        num_hits,
        avg_reward,
        exact_hit_rate,
        soft_hit_rate,
        exact_ep_correct,
        soft_ep_correct,
        avg_soft_ep_correct,
        train_loss,
        optimizer_steps,
    ):
        print(
            "[Train] Step {0:4d}/{1:4d} ({2:6.2f}%) | Remaining: {3:4d} | "
            "Total Reward: {4:7.4f} | Avg Reward: {5:7.4f} | "
            "Exact Hit: {6:7.4f} | Soft Hit: {7:7.4f} | "
            "Exact EP Success: {8:4d} | Soft EP Success: {9:4d} | "
            "Soft EP Rate: {10:7.4f} | Loss: {11:7.4f} | Optimizer Steps: {12:4d}".format(
                step,
                total_steps,
                progress_pct,
                remaining_steps,
                num_hits,
                avg_reward,
                exact_hit_rate,
                soft_hit_rate,
                exact_ep_correct,
                soft_ep_correct,
                avg_soft_ep_correct,
                train_loss,
                optimizer_steps,
            )
        )

    @staticmethod
    def log_eval_start(step):
        print("[Eval] Starting periodic beam evaluation at step {} ...".format(step))

    @staticmethod
    def log_eval_end(step):
        print("[Eval] Finished periodic evaluation at step {}.".format(step))

    @staticmethod
    def log_eval_config(beam, path_length, test_rollouts, total_examples):
        print(
            "[Eval] Running evaluation | beam={} | path_length={} | test_rollouts={} | total_examples={}".format(
                beam,
                path_length,
                test_rollouts,
                total_examples,
            )
        )

    @staticmethod
    def log_eval_completed():
        print("[Eval] Evaluation completed.")

    def export_raw_paths(self, path_logger_file_prefix, paths, answers):
        print("[Eval] Printing verbose path traces at {}".format(self.output_dir + "/test_beam/"))
        for q in paths:
            j = q.replace("/", "-")
            output_path = path_logger_file_prefix + "_" + j
            with open(output_path, "a", encoding="utf-8") as pos_file:
                for p in paths[q]:
                    pos_file.write(p)
        with open(path_logger_file_prefix + "answers", "w", encoding="utf-8") as answer_file:
            for a in answers:
                answer_file.write(a)

    @staticmethod
    def export_clean_summary(path_logger_file_prefix, clean_export_blocks):
        clean_export_file = path_logger_file_prefix + "_clean_summary.txt"
        with open(clean_export_file, "w", encoding="utf-8") as clean_file:
            clean_file.write("\n".join(clean_export_blocks) + "\n")
        print("[Eval] Wrote cleaner beam summary to {}".format(clean_export_file))

    def append_verbose_query_block(
        self,
        paths,
        answers,
        query_relation_name,
        start_entity_name,
        end_entity_name,
        sorted_indices_row,
        query_batch_index,
        test_rollouts,
        rewards,
        positive_reward,
        se,
        ce,
        log_probs,
        entity_trajectory,
        relation_trajectory,
        action_type_trajectory,
        id2entity,
        id2relation,
        answer_pos,
    ):
        paths[str(query_relation_name)].append("Start: {}\t Target: {}\n".format(start_entity_name, end_entity_name))
        paths[str(query_relation_name)].append(
            "Reward: {}\n".format(1 if answer_pos is not None and answer_pos < 10 else 0)
        )

        for r in sorted_indices_row:
            global_index = query_batch_index * test_rollouts + r
            rev = 1 if rewards[global_index] == positive_reward else -1

            answers.append(
                self._safe_lookup(id2entity, se[query_batch_index, r])
                + "\t"
                + self._safe_lookup(id2entity, ce[query_batch_index, r])
                + "\t"
                + str(log_probs[query_batch_index, r])
                + "\n"
            )

            entity_trace = "\t".join([self._safe_lookup(id2entity, e[global_index]) for e in entity_trajectory])
            relation_trace = "\t".join([self._safe_lookup(id2relation, re[global_index]) for re in relation_trajectory])
            action_type_trace = "\t".join([str(step_types[global_index]) for step_types in action_type_trajectory])

            paths[str(query_relation_name)].append(
                entity_trace
                + "\n"
                + relation_trace
                + "\n"
                + "Hop Types: "
                + action_type_trace
                + "\n"
                + str(rev)
                + "\n"
                + str(log_probs[query_batch_index, r])
                + "\n___\n"
            )

        paths[str(query_relation_name)].append("#####################\n")

    def append_clean_summary_block(
        self,
        clean_export_blocks,
        query_relation_name,
        start_entity_id,
        end_entity_id,
        answer_pos,
        sorted_indices_row,
        ce_row,
        log_probs_row,
        top_k,
        id2entity,
    ):
        start_entity_name = self._safe_lookup(id2entity, start_entity_id)
        end_entity_name = self._safe_lookup(id2entity, end_entity_id)

        ranked_unique = []
        seen_entities_for_export = set()
        for r in sorted_indices_row:
            entity_id = int(ce_row[r])
            if entity_id in seen_entities_for_export:
                continue
            seen_entities_for_export.add(entity_id)
            ranked_unique.append((entity_id, float(log_probs_row[r])))
            if len(ranked_unique) >= top_k:
                break

        best_correct_rank = None
        for rank_idx, (pred_entity_id, _) in enumerate(ranked_unique, start=1):
            if pred_entity_id == int(end_entity_id):
                best_correct_rank = rank_idx
                break

        clean_export_blocks.append(
            "Query Relation: {}\nStart Entity: {} ({})\nTarget Entity: {} ({})\nHit@10: {}\nBest Correct Rank: {}\nTop {} Unique Predictions:".format(
                query_relation_name,
                start_entity_name,
                int(start_entity_id),
                end_entity_name,
                int(end_entity_id),
                "YES" if (answer_pos is not None and answer_pos < 10) else "NO",
                best_correct_rank if best_correct_rank is not None else "Not found in top {}".format(top_k),
                int(top_k),
            )
        )
        for rank_idx, (pred_entity_id, pred_score) in enumerate(ranked_unique, start=1):
            clean_export_blocks.append(
                "  {}. {} ({}) | score={:.6f} | correct={}".format(
                    rank_idx,
                    self._safe_lookup(id2entity, pred_entity_id),
                    pred_entity_id,
                    pred_score,
                    "YES" if pred_entity_id == int(end_entity_id) else "NO",
                )
            )
        clean_export_blocks.append("-" * 80)
