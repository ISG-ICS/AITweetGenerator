# AITweetGenerator by Tiancheng (Joseph) Zheng

## Description
An AI based tweet generator which can generate collection of tweets automatically. It can be used to replace the usernames and texts of the real twitter data provided by the user with fake generated usernames and texts. The tweets are generated using models learned from real tweets using the `textgenrnn` library.

## Prerequisites
### Libraries
- `textgenrnn`, which can be installed using `pip install textgenrnn`
- `ujson`, which can be installed using `pip install ujson`
- `tensorflow`, which can be installed using `pip install tensorflow`
- If other libraries are missing, install them using `pip install <library>`
### Data
Users must provide the raw twitter data downloaded from Twitter and unzip it to a json file. This json file will be fed to both the trainer and the generator.
The downloaded Twitter data is in `.gz` format (e.g. `Tweet_2019-07-15_08-17-17.gz`). To unzip the data to a json file, use command `gunzip -c <filename>.gz > <filename>.json` (e.g. `gunzip -c Tweet_2019-07-15_08-17-17.gz > tweets.json`).
### Python Version
Python version must be at least Python 3.6.

## Pipeline
0. IMPORTANT: create a subdirectory called `weights` in the same directory as the scripts for the pipeline to work.
1. Run `tweet_trainer.py` to train a model using the `textgenrnn` library. It trains the model using a sample of size `k` from the json file containing the raw twitter data, which is provided by the user. It takes three parameters: `-i` -- the json file storing the original tweets (e.g. `tweets.json`), `-k` -- the training sample size, and `-e` -- the training epoch. An example is `python3 tweet_trainer.py -i tweets.json -k 20000 -e 5`. Please use `python3 tweet_trainer.py --help` for a detailed description of the script and its parameters.
2. Run `tweet_generator.py` to replace tweets with fake usernames and texts. The fake usernames are generated by the helper function `name_generator.py`, and the fake texts are generated using the trained model. The script generates the tweets and writes them into another json file in batches of size `k`, which is specified by the user. It takes three parameters: `-i` -- the json file storing the original tweets (e.g. `tweets.json`),  `-o` -- the json file to store the fake tweets (e.g. `fake_tweets.json`), and `-k` -- the batch size to generate. An example is `python3 text_generator.py -i tweets.json -o fake_tweets.json -k 12000`. Please use `python3 text_generator.py --help` for a detailed description of the script and its parameters.

## Appendix
### Relevant Scripts
1. name_generator.py
2. tweet_trainer.py
3. tweet_generator.py
