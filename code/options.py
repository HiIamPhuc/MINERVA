from __future__ import absolute_import
from __future__ import division
import argparse
import uuid
import os
from pprint import pprint


def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="", type=str, required=True)
    parser.add_argument("--output_dir", default='./output/', type=str)
    parser.add_argument("--log_dir", default="./logs/", type=str)

    parser.add_argument("--max_actions", default=400, type=int)
    parser.add_argument("--path_length", default=3, type=int)
    parser.add_argument("--total_steps", default=2000, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--grad_clip_norm", default=5, type=int)
    parser.add_argument("--l2_reg_const", default=1e-2, type=float)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--entropy_weight", default=1e-2, type=float)
    parser.add_argument("--positive_reward", default=1.0, type=float)
    parser.add_argument("--negative_reward", default=0, type=float)
    parser.add_argument("--discount_factor", default=1, type=float)
    parser.add_argument("--num_rollouts", default=20, type=int)
    parser.add_argument("--test_rollouts", default=100, type=int)
    parser.add_argument("--num_lstm_layers", default=1, type=int)
    parser.add_argument("--hidden_size", default=50, type=int)

    parser.add_argument("--baseline_decay", default=0.0, type=float)
    parser.add_argument("--eval_pool_mode", default="max", type=str)
    parser.add_argument("--eval_interval", default=100, type=int)
    parser.add_argument("--nell_evaluation", default=0, type=int)

    parser.add_argument("--gwm_model_path", default="", type=str, help="Path to pre-trained GWM model checkpoint")
    parser.add_argument("--hallucinate_k", default=3, type=int, help="Number of latent edges to hallucinate")

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))

    parsed['model_dir'] = parsed['output_dir'] + '/' + 'model/'
    parsed['log_file_name'] = parsed['output_dir'] +'/log.txt'

    os.makedirs(parsed['output_dir'], exist_ok=True)
    os.makedirs(parsed['model_dir'], exist_ok=True)
    with open(parsed['output_dir']+'/config.txt', 'w') as out:
        pprint(parsed, stream=out)

    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()):
        print(fmtString % keyPair)

    return parsed