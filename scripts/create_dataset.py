import argparse
import random
import os.path as p


def main():
    parser = argparse.ArgumentParser(description='restore a othello game')
    parser.add_argument('SOURCE', help='path to source data')
    parser.add_argument('--train-size', type=int, help='path to target data')
    parser.add_argument('--valid-size', type=int, help='path to target data')
    args = parser.parse_args()

    with open(args.SOURCE, "r") as sou:
        source = [line.strip() for line in sou]
        assert len(source) > args.train_size, 'train size exceeds input data size'
        dir_name = p.dirname(args.SOURCE)

    random.shuffle(source)
    train = [line for line in source[:args.train_size]]
    ref = set(train)
    target = [line for line in source[args.train_size:] if line not in ref]
    assert len(target) >= args.valid_size, 'could not retrieve valid data due to too large valid size specified'
    valid = target[:args.valid_size]

    train_path = p.join(dir_name, 'atrain.txt')
    valid_path = p.join(dir_name, 'avalid.txt')

    with open(train_path, "w") as tra:
        for line in train:
            tra.write(line+'\n')

    with open(valid_path, "w") as val:
        for line in valid:
            val.write(line+'\n')


if __name__ == '__main__':
    main()
