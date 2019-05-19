# Othello AI

Predict next move from a board.

## Data Preparation
1. Download records of an Othello game from the following URL

    https://www.skatgame.net/mburo/ggs/game-archive/Othello/

2. Cleanse the data
    ```
    $ python scripts/cleanse_data.py data/hoge data/cleansed.txt
    ```

3. Extract board and move pairs.
    ```
    $ python scripts/restore_from_data.py data/cleansed.txt data/restored.txt
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

This script only extracts the winner's actions to select one from more than two valid moves.

### Create Dataset
Run the following command, and train/validation data will be created in the same directory as the input file.
```
$ python scripts/create_dataset.py data/restored.txt --train-size  100 --valid-size 10
```
You may fail to create dataset if too large size is specified.

## Environment
```
$ pip install pipenv --user
$ pipenv install
$ pipenv shell
```

## Model
In progress ...

Now supervised-learning policy network is available
```
$ python src/train.py config/sample.json [**kwargs]
```    

## Result
|data|size|
| --- | ---: |
|Train|10,000,000 pairs|
|Valid|624,169 pairs|

(no duplicates between train and dev)

| model | max validation accuracy |
| --- | ---: |
| supervised-learning policy network ||
