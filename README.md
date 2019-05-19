# Othello AI

Predict next move from a board.

### Data Preparation
1. Download records of an Othello game from the following URL

    https://www.skatgame.net/mburo/ggs/game-archive/Othello/

2. Cleanse the data
    ```
    $ python script/cleanse_data.py data/hoge data/cleansed.txt
    ```

3. Extract board and move pairs.
    ```
    $ python script/restore_from_data.py data/cleansed.txt data/restored.txt
    ```

You will see these lines in restored.txt.

    > 0000000000000000000100000001100000010000000000000000000000000000 0000000000000000000000000000000000001000000000000000000000000000 W 18
    > 0000000000000000000100000011100000010000000000000000000000000000 0000000000000000001000000000000000001000000000000000000000000000 W 20
    > ...
   
Each column shows
1. where black discs are placed (expressed by 64-bits number)
2. where white discs are placed (expressed by 64-bits number)
3. the color of a player who makes a next move
4. next move (expressed by an integer from -1 to 63)

This script only extracts the winner's actions to select one from at least two valid moves.

### Model
In Progress

### Result
|data|size|
| --- | ---: |
|Train|10,000,000 pairs|
|Dev|624,169 pairs|

(no duplicates between train and dev)

| model | max validation accuracy |
| --- | ---: |
| only supervised-learning policy network |  ---|
