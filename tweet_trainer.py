import tensorflow as tf
import os
import re
from textgenrnn import textgenrnn
import time
import click
import ujson as json


def process_tweet_text(text):
    text = re.sub(r'http\S+', '', text)   # Remove URLs
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)  # Remove @ mentions
    text = text.strip(" ")   # Remove whitespace resulting from above
    text = re.sub(r' +', ' ', text)   # Remove redundant spaces
    
    # Handle common HTML entities
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&amp;', '&', text)
    return text


def train_model(infile, size, epoch):
    cfg = {'num_epochs': epoch,
           'gen_epochs': 1,
           'batch_size': 128,
           'train_size': 1.0,
           'new_model': False,
           'model_config': {'rnn_layers': 2,
                            'rnn_size': 128,
                            'rnn_bidirectional': False,
                            'max_length': 40,
                            'dim_embeddings': 100,
                            'word_level': False
                            }
           }
    texts = []
    context_labels = []

    print('Loading training sample from file...')
    start_time = time.time()
    with open(infile, 'r') as f:
        for line in f:
            try:
                s = json.loads(line)
                if 'text' in s.keys():
                    tweet_text = process_tweet_text(s['text'])
                    if tweet_text is not '':
                        texts.append(tweet_text)
                        context_labels.append(s['user']['screen_name'])
                        if len(texts) == size:
                            break
            except ValueError:
                print('Reached end of file!')
    print("Load time: {} seconds".format(time.time() - start_time))

    print('Actual sample size:', len(texts))

    textgen = textgenrnn(name='./weights/twitter_general')
    
    if cfg['new_model']:
        textgen.train_new_model(
            texts,
            context_labels=context_labels,
            num_epochs=cfg['num_epochs'],
            gen_epochs=cfg['gen_epochs'],
            batch_size=cfg['batch_size'],
            train_size=cfg['train_size'],
            rnn_layers=cfg['model_config']['rnn_layers'],
            rnn_size=cfg['model_config']['rnn_size'],
            rnn_bidirectional=cfg['model_config']['rnn_bidirectional'],
            max_length=cfg['model_config']['max_length'],
            dim_embeddings=cfg['model_config']['dim_embeddings'],
            word_level=cfg['model_config']['word_level'])
    else:
        textgen.train_on_texts(
            texts,
            context_labels=context_labels,
            num_epochs=cfg['num_epochs'],
            gen_epochs=cfg['gen_epochs'],
            train_size=cfg['train_size'],
            batch_size=cfg['batch_size'])


@click.command()
@click.option('--infile', '-i',
              required=True,
              help='Enter the json file storing the original tweets (e.g. tweets.json).')
@click.option('--size', '-k', type=click.INT,
              required=True,
              help='Enter the training sample size.')
@click.option('--epoch', '-e', type=click.INT,
              required=True,
              help='Enter the training epoch.')
def main(infile, size, epoch):
    # silence tensorflow
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # training general tweets
    print('Training general tweets with sample size k = {}...'.format(size))
    start_time = time.time()
    try:
        train_model(infile, size, epoch)
    except ValueError:
        pass
    print("Training time: {} seconds".format(time.time() - start_time))


if __name__ == '__main__':
    main()
