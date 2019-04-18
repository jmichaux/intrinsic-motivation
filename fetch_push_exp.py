import argparse
import itertools
import yaml

import numpy as np

parser = argparse.ArgumentParser(description='PPO on FetchReach-v1 with shared optimization of policy and value function.')
parser.add_argument('--experiment-name', nargs='+', type=str,
                    default=["RandomAgent"])
parser.add_argument('--env-id', nargs='+', type=str,
                    default=["FetchPush-v1"])
parser.add_argument('--hidden-size', nargs='+', type=int,
                    default=[64, 128])
parser.add_argument('--pi-lr', nargs='+',type=float, help='learning rate for the policy parameter',
                    default=[1e-4, 5e-4, 1e-3])
parser.add_argument('--clip-param', nargs='+', type=float, help='ppo clipping parameter',
                    default=[0.35, 0.2, 0.1])
parser.add_argument('--value-coef', nargs='+', type=float, help='value loss coefficient',
                    default=[0.5, 0.25, 0.125])
parser.add_argument('--entropy-coef', nargs='+', type=float, help='entropy loss coefficient',
                    default=[0, 0.01, 0.1])
parser.add_argument('--grad-norm-max', nargs='+', type=float, help='clip max norm of the gradient',
                    default=[0.5, 1.0])
parser.add_argument('--gamma', nargs='+', type=float, help='discount factor',
                    default=[0.99])
parser.add_argument('--gae-lambda', nargs='+', type=float, help='generalized advantage estimation parameters',
                    default=[0.95])
# parser.add_argument('--num_instances', type=int, default=4)

num_instances = 4

# parse arguments
args = parser.parse_args()

# dictionaries are now ordered by default in Python >=3.6
args_dict = args.__dict__
arg_list = []
for key in list(args.__dict__.keys()):
    if args_dict[key] is None:
        continue
    arg_list.append(args_dict[key])

# command template
template = "python main.py --experiment-name {0} --env-id {1} --num-env-steps 20000000 --num-processes 64 --num-steps 2048 --ppo-epoch 10 --num-mini-batch 32 --hidden-size {2} --pi-lr {3} --clip-param {4} --value-coef {5} --entropy-coef {6} --grad-norm-max {7} --gamma {8} --gae-lambda {9} --use-gae --share-optim  --use-clipped-value-loss --log-interval 1 --use-tensorboard"
cmd = ["source launch_random_agent.sh"]

config = {"session_name": "run-all", "windows": []}

experiments = []
for element in itertools.product(*arg_list):
        experiments.append(template.format(*element))

experiments = [list(exp) for exp in np.array_split(experiments, num_instances)]

for i, exp in enumerate(experiments):
    with open('launch_{}_{}_{}.sh'.format('fetch_reach_v1', 'random_agent', i+1), 'w') as f:
        for item in exp:
            f.write("{}\n".format(item))
# config["windows"].append({"window_name": "{}/{}".format(args.env_id[0], args.experiment_name[0]), "panes": experiments})
# yaml.dump(config, open("run_random_agent.yaml", "w"), default_flow_style=False)
