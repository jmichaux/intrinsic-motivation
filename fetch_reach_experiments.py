import argparse
import itertools
import yaml

parser = argparse.ArgumentParser(description='PPO on FetchReach-v1 with shared optimization of policy and value function.')
parser.add_argument('--experiment-name', nargs='+', type=str,
                    default=["RandomAgent"])
parser.add_argument('--env-id', nargs='+', type=str,
                    default=["FetchReach-v1"])
parser.add_argument('--hidden-size', nargs='+', type=int,
                    default=[64])
parser.add_argument('--pi-lr', nargs='+',type=float, help='learning rate for the policy parameter',
                    default=[1e-4, 5e-4, 1e-3])
parser.add_argument('--clip-param', nargs='+', type=float, help='ppo clipping parameter',
                    default=[0.05, 0.1, 0.2])
parser.add_argument('--value-coef', nargs='+', type=float, help='value loss coefficient',
                    default=[0.125, 0.25, 0.5])
parser.add_argument('--entropy-coef', nargs='+', type=float, help='entropy loss coefficient',
                    default=[0, 0.01, 0.05, 0.1])
parser.add_argument('--grad-norm-max', nargs='+', type=float, help='clip max norm of the gradient',
                    default=[0.5, 1.0])
parser.add_argument('--gamma', nargs='+', type=float, help='discount factor',
                    default=[0.99])
parser.add_argument('--gae-lambda', nargs='+', type=float, help='generalized advantage estimation parameters',
                    default=[0.95])


# parse arguments
args = parser.parse_args()

# dictionaries are now ordered by default in Python >=3.6
args_dict = args.__dict__
ordered_keys = list(args.__dict__.keys())
arg_list = []
for key in ordered_keys:
    if args_dict[key] is None:
        continue
    arg_list.append(args_dict[key])

# command template
template = "python main.py --experiment-name {0} --env-id {1} --num-env-steps 100000 --num-processes 1 --num-steps 2048 --ppo-epoch 10 --num-mini-batch 32 --hidden-size {2} --pi-lr {3} --clip-param {4} --value-coef {5} --entropy-coef {6} --grad-norm-max {7} --gamma {8} --gae-lambda {9} --use-gae --share-optim  --use-clipped-value-loss --log-interval 1 --use-tensorboard"
cmd = ["source launch_random_agent.sh"]

config = {"session_name": "run-all", "windows": []}

experiments = []
for element in itertools.product(*arg_list):
        experiments.append(template.format(*element))
with open('launch_random_agents.sh', 'w') as f:
    for item in experiments:
        f.write("{}\n".format(item))
config["windows"].append({"window_name": "{}/{}".format(args.env_id[0], args.experiment_name[0]), "panes": experiments})
yaml.dump(config, open("run_random_agent.yaml", "w"), default_flow_style=False)
