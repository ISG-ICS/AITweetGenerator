import tensorflow as tf
import os
from textgenrnn import textgenrnn
import multiprocessing as mp
import time
import itertools
import ujson as json
import click
import numpy as np
import name_generator as ng


def generate(k):
    np.random.seed()
    textgen = textgenrnn('./weights/twitter_general_weights.hdf5')
    text = textgen.generate(n=k, max_gen_length=140, return_as_list=True)
    return text


@click.command()
@click.option('--infile', '-i',
              required=True,
              help='Enter the json file storing the original tweets (e.g. tweets.json).')
@click.option('--outfile', '-o',
              required=True,
              help='Enter the json file to store the fake tweets (e.g. fake_tweets.json).')
@click.option('--size', '-k', type=click.INT,
              required=True,
              help='Enter the batch size to generate.')
def main(infile, outfile, size):
    # silence tensorflow
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    print('Generating general fake texts...')
    start_time = time.time()
    name_dict = {}
    count = 0
    with open(infile, 'r') as fin, open(outfile, 'w') as fout:
        flag = True
        while flag:
            tweet = []
            while len(tweet) < size:
                try:
                    s = json.loads(next(fin))
                    if 'text' in s.keys():
                        tweet.append(s)
                except ValueError:
                    print('Reached end of file!')
                    flag = False
                    break
            count += len(tweet)
            if len(tweet) % mp.cpu_count() == 0:
                k = len(tweet) // mp.cpu_count()
            else:
                k = len(tweet) // mp.cpu_count() + 1
            pool = mp.Pool(mp.cpu_count())
            ftext = pool.map(generate, [k for _ in range(mp.cpu_count())])
            pool.close()
            ftext = list(itertools.chain.from_iterable(ftext))
            for s, t in zip(tweet, ftext[:len(tweet)]):
                s['text'] = t
                if s['user']['id'] not in name_dict.keys():
                    id = s['user']['id']
                    name = ng.gen_two_words(split=' ', lowercase=False)
                    screenname = ''.join(name.split(' ')).lower() + ng.gen_year(1900, 2020) + ng.gen_birthday()
                    name_dict[id] = {'name': name, 'screen_name': screenname}
                    s['user']['name'] = name
                    s['user']['screen_name'] = screenname
                else:
                    id = s['user']['id']
                    s['user']['name'] = name_dict[id]['name']
                    s['user']['screen_name'] = name_dict[id]['screen_name']
                line = json.dumps(s)
                fout.write(line + '\n')
            print('Processed records:', count)

    print('Generate time: {} seconds'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
