import argparse


def main():
    parser = argparse.ArgumentParser(description='restore a othello game')
    parser.add_argument('SOURCE', help='path to source data')
    parser.add_argument('TARGET', help='path to target data')
    parser.add_argument('OUTPUT', help='path to output')
    args = parser.parse_args()

    with open(args.SOURCE, "r") as sou, open(args.TARGET, "r") as tar:
        source = set([line.strip() for line in sou])
        target = [line.strip() for line in tar]

    count = 0
    with open(args.OUTPUT, "w") as out:
        for line in target:
            if line not in source:
                out.write(line + '\n')
                count += 1
        else:
            print(count)


if __name__ == '__main__':
    main()
