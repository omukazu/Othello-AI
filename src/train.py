import argparse
import json
from argparse import RawTextHelpFormatter

import chainer
from chainer import iterators, training
from chainer.backends import cuda
from chainer.training import extensions
import numpy as np

from sl_policy_network import SLPolicyNetwork
from utils import load_data, print_board


def main():
    parser = argparse.ArgumentParser(description='SLPolicyNetwork', formatter_class=RawTextHelpFormatter)
    parser.add_argument('CONFIG', default=None, type=str, help='path to config file')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu numbers\nto specify')
    parser.add_argument('--debug', default=False, action='store_true', help='switch to debug mode')
    args = parser.parse_args()

    with open(args.CONFIG, "r") as f:
        config = json.load(f)

    path = 'debug' if args.debug else 'data'

    b = config["arguments"]["batch_size"]
    epoch = config["arguments"]["epoch"]

    print('*** making training data ***')
    train_data = load_data(config[path]["train"])  # (state, action) = ((3, 8, 8), (1))
    train_iter = iterators.SerialIterator(train_data, b)

    valid_data = load_data(config[path]["valid"])
    valid_iter = iterators.SerialIterator(valid_data, b, repeat=False, shuffle=False)

    print('*** preparing model ***')
    n_input_channel = config["arguments"]["n_input_channel"]
    n_output_channel = config["arguments"]["n_output_channel"]
    model = SLPolicyNetwork(n_input_channel=n_input_channel,
                            n_output_channel=n_output_channel)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)
    model._set_cache()

    optimizer = chainer.optimizers.Adam(alpha=config["arguments"]["learning_rate"])
    optimizer.setup(model)

    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu)

    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result' + '/' + config["arguments"]["save_path"])

    @chainer.training.make_extension()
    def predict_next_move(_):
        state, action = valid_data[np.random.choice(len(valid_data))]
        n_channel, row, column = state.shape
        if args.gpu >= 0:
            state = cuda.to_gpu(state)
        prediction = model.predict(state.reshape(1, n_channel, row, column))
        if args.gpu >= 0:
            state = cuda.to_cpu(state)
        print_board(state)
        print(f'action : {action}')
        print(f'prediction : {prediction}')

    @chainer.training.make_extension()
    def print_iter(_):
        print(f'*** 100 iterate ***')

    trainer.extend(predict_next_move, trigger=(1, 'epoch'))
    trainer.extend(print_iter, trigger=(100, 'iteration'))

    trainer.extend(extensions.Evaluator(valid_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy',
                                           'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.snapshot_object(model, 'slpn.epoch{.updater.epoch}.npz'), trigger=(10, 'epoch'))
    save_trigger_for_accuracy = chainer.training.triggers.MaxValueTrigger(key='validation/main/accuracy',
                                                                          trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, 'slpn.best_accuracy.npz'), trigger=save_trigger_for_accuracy)
    save_trigger_for_loss = chainer.training.triggers.MinValueTrigger(key='validation/main/loss',
                                                                      trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, 'slpn.best_loss.npz'), trigger=save_trigger_for_loss)

    print('*** start training ***')
    trainer.run()


if __name__ == '__main__':
    main()
