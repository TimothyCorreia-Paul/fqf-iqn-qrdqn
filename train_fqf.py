import os
import yaml
import argparse
from datetime import datetime

from fqf_iqn_qrdqn.env import make_pytorch_env
from fqf_iqn_qrdqn.agent import FQFAgent


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = make_pytorch_env(env_id=args.env_id, frameskip=args.frameskip, repeat_action_probability=args.repeat_action_probability)
    test_env = make_pytorch_env(
        env_id=args.env_id, frameskip=args.frameskip, repeat_action_probability=args.repeat_action_probability, episode_life=False, clip_rewards=False)

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')

    # Create the agent and run.
    agent = FQFAgent(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'fqf.yaml'))
    parser.add_argument('--env_id', type=str, default='ALE/MsPacman-v5')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--frameskip', type=tuple, default=(2, 5,))
    parser.add_argument('--repeat_action_probability', type=float, default=0.0)
    args = parser.parse_args()
    if len(args.frameskip) == 1:
        args.frameskip = args.frameskip[0]
        assert type(args.frameskip) == int
    run(args)
