import argparse


def valid_moves(moves):
    row = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'}
    column = {'1', '2', '3', '4', '5', '6', '7', '8'}
    for move in moves:
        if (move == 'PA') or (move[0] in row and move[1] in column):
            continue
        else:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description='cleanse data')
    parser.add_argument('INPUT', help='path to input data')
    parser.add_argument('OUTPUT', help='path to output data of states')
    args = parser.parse_args()

    """ cleanse data which contains records of an Othello game
    
    data source ... https://www.skatgame.net/mburo/ggs/game-archive/Othello/?C=N;O=A
    the details of INPUT data format ... https://skatgame.net/mburo/ggs/ggf
    """

    initial_board = 'BO[8 -------- -------- -------- ---O*--- ---*O--- -------- -------- -------- *]'

    # ['{n_game} (;GM ... W[G8];)', ... ,] -> ['GM ... W[G8];)', ... , ], n_game = {1, 2}
    lines = []
    with open(args.INPUT, "r") as inp:
        for line in inp:
            n_game = int(line[0])
            if n_game == 1:
                lines.append(line.strip().split('(;')[1])
            elif n_game == 2:
                lines.extend(line.strip().split('(;')[1:])
            else:
                raise Exception
    assert all([line.startswith('GM') for line in lines]), 'invalid input'

    """
    ['GM ... W[G8];)', ... , ] -> ['RE[+2.000] ... W[G8]', ... , ]
    -> ['RE[+2.000]BO[ ... W[G8]', ... , ] -> [['RE[+2.000', 'B[C4' ... , 'W[G8'], ... , ]
    -> [['RE[+2.000', 'B[C4', ... , 'W[G8'], ... , ] -> [['2', 'C4', ... , 'G8'], ... , ]
    -> [['2', 'C4', ... , 'G8'], ... , ] -> [['2', 'C4', ... , 'G8'], ... , ]

    remove data does not start with initial state, contains flag(r,s,t) or invalid moves, and results in draw
    """
    lines = [line[line.find('RE'):-2] for line in lines if initial_board in line]
    lines = [[e.split('/')[0] for i, e in enumerate(line.split(']')[:-1]) if i != 1]
             for line in lines if ':' not in line]
    lines = [[int(float(e.split('[')[1])) if i == 0 else e.split('[')[1][:2].upper() for i, e in enumerate(line)]
             for line in lines]
    lines = [line for line in lines if line[0] != 0 and valid_moves(line[1:])]
    assert len(set([move for line in lines for move in line[1:]])) == 61, 'still contains invalid moves'

    with open(args.OUTPUT, "w") as out:
        for line in lines:
            winner = 'B' if line[0] > 0 else 'W'
            string = winner + ' ' + ' '.join(line[1:]) + '\n'
            out.write(string)


if __name__ == '__main__':
    main()
