from __future__ import absolute_import
from __future__ import division
import argparse
import uuid
import os
import yaml
from pprint import pprint


class Config(dict):
    """Dictionary-like config with attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    @classmethod
    def from_dict(cls, data):
        return cls(data)


def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to MINERVA yaml config")

    # Optional CLI overrides for rapid experiments
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--gwm_model_path", type=str, default=None)
    parser.add_argument("--hallucinate_k", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)

    try:
        cli_args = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))

    config_path = cli_args.pop("config")
    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        parser.error("YAML config must parse into a dictionary")

    defaults = {
        "data_dir": "",
        "output_dir": "",
        "log_dir": "./logs",
        "max_actions": 400,
        "path_length": 3,
        "total_steps": 2000,
        "batch_size": 128,
        "grad_clip_norm": 5,
        "l2_reg_const": 1e-2,
        "learning_rate": 1e-3,
        "entropy_weight": 1e-2,
        "positive_reward": 1.0,
        "negative_reward": 0.0,
        "discount_factor": 1.0,
        "num_rollouts": 20,
        "test_rollouts": 100,
        "num_lstm_layers": 1,
        "hidden_size": 50,
        "baseline_decay": 0.0,
        "eval_pool_mode": "max",
        "eval_interval": 100,
        "nell_evaluation": 0,
        "gwm_model_path": "",
        "hallucinate_k": 3,
        "gwm_text_cache_device": "cpu",
    }

    cfg = {**defaults, **loaded}

    # CLI overrides always win when provided
    for k, v in cli_args.items():
        if v is not None:
            cfg[k] = v

    if not cfg["data_dir"]:
        parser.error("data_dir must be provided in YAML or via --data_dir")

    if not cfg.get("output_dir"):
        run_name = "minerva_" + str(uuid.uuid4())[:8]
        cfg["output_dir"] = os.path.join(cfg["log_dir"], run_name)

    cfg["model_dir"] = os.path.join(cfg["output_dir"], "model")
    cfg["path_logger_file"] = cfg["output_dir"]
    cfg["log_file_name"] = os.path.join(cfg["output_dir"], "log.txt")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["model_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "resolved_config.yaml"), "w", encoding="utf-8") as out_yaml:
        yaml.safe_dump(cfg, out_yaml, sort_keys=True)
    with open(os.path.join(cfg["output_dir"], "config.txt"), "w", encoding="utf-8") as out_txt:
        pprint(cfg, stream=out_txt)

    maxLen = max([len(ii) for ii in cfg.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(cfg.items()):
        print(fmtString % keyPair)

    return Config.from_dict(cfg)