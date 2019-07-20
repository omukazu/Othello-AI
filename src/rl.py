import argparse
import glob
import json
import time
from argparse import RawTextHelpFormatter

import chainer
import numpy as np
from chainer import serializers
from chainer.backends import cuda
from progressbar import ProgressBar

from agent import Agent
from env import Env
from sl_policy_network import SLPolicyNetwork
from rl_policy_network import RLPolicyNetwork


def main():
    parser = argparse.ArgumentParser(description='SLPolicyNetwork', formatter_class=RawTextHelpFormatter)
    parser.add_argument('CONFIG', default=None, type=str, help='path to config file')
    parser.add_argument('MODEL', default=None, type=str, help='path to model.npz')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu numbers\nto specify')
    parser.add_argument('--debug', default=False, action='store_true', help='switch to debug mode')
    args = parser.parse_args()

    print('*** load config ***')
    with open(args.CONFIG, "r") as f:
        config = json.load(f)

    print('*** set up model ***')
    n_input_channel = config["arguments"]["n_input_channel"]
    n_output_channel = config["arguments"]["n_output_channel"]
    rl_policy_network = RLPolicyNetwork(n_input_channel=n_input_channel, n_output_channel=n_output_channel)
    serializers.load_npz(args.MODEL, rl_policy_network)
    optimizer = chainer.optimizers.Adam(alpha=config["arguments"]["learning_rate"])
    optimizer.setup(rl_policy_network)

    if args.gpu:
        cuda.get_device_from_id(args.gpu).use()
        rl_policy_network.to_gpu(args.gpu)
        xp = cuda.cupy
    else:
        xp = np
    rl_policy_network.set_cache()

    # define parameters
    N = 30000
    batch_size = 150
    first_choices = [0x0000100000000000, 0x0000002000000000, 0x0000000004000000, 0x0000000000080000]
    bar = ProgressBar(0, N)

    print('*** start iteration ***')
    for i in range(N):
        bar.update(i)
        start = time.time()
        opponent = SLPolicyNetwork(n_input_channel=n_input_channel, n_output_channel=n_output_channel)
        opponent_model_path = np.random.choice(glob.glob("./result/sl_policy/slpn.epoch*.npz"))
        print(f'\nmodel:{opponent_model_path} is chosen')
        serializers.load_npz(opponent_model_path, opponent)
        if args.gpu:
            opponent.to_gpu(args.gpu)
        opponent.set_cache()
        agent = Agent(batch_size, xp, rl_policy_network, optimizer)
        env = Env(batch_size, xp, rl_policy_network, opponent)
        env.reset()

        is_black = True
        if i % 2 == 1:
            first_actions = xp.random.choice(first_choices, batch_size).astype('uint64').reshape(-1, 1)
            reversible_mask = env.reversible(first_actions, is_black)
            env.black, env.white = \
                env.reverse(first_actions, is_black, reversible_mask)
            is_black = not is_black

        obs = env.create_current_states(is_black)
        done = False
        while not done:
            action_indices = agent.act(obs)
            obs, _, done, _ = env.step(action_indices, is_black)

        bs = xp.sum(obs[:, 0].reshape(batch_size, -1), axis=1)  # (b, 8, 8) -> (b, )
        ws = xp.sum(obs[:, 1].reshape(batch_size, -1), axis=1)

        true_rewards = bs > ws if is_black else ws > bs
        agent.update(true_rewards)

        count = xp.sum(bs > ws) if is_black else xp.sum(ws > bs)
        print(f'{time.time() - start:.02f} sec elapsed')
        print(f'win rate:{int(count) * 100 / batch_size:.02f}')
    else:
        serializers.save_npz("result/rl_policy.npz", rl_policy_network)


if __name__ == '__main__':
    main()