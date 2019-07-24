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
    # generate tweets
    print('Generating general fake tweets...')
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
                except (StopIteration, ValueError) as error:
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
                uid = s['user']['id']
                if uid not in name_dict.keys():
                    name = ng.gen_two_words(split=' ', lowercase=False)
                    screen_name = ''.join(name.split(' ')).lower() + ng.gen_year(1900, 2020) + ng.gen_birthday()
                    name_dict[uid] = {'name': name, 'screen_name': screen_name}
                    s['user']['name'] = name
                    s['user']['screen_name'] = screen_name
                else:
                    s['user']['name'] = name_dict[uid]['name']
                    s['user']['screen_name'] = name_dict[uid]['screen_name']
                for i in range(len(s['entities']['user_mentions'])):
                    mid = s['entities']['user_mentions'][i]['id']
                    if mid not in name_dict.keys():
                        name = ng.gen_two_words(split=' ', lowercase=False)
                        screen_name = ''.join(name.split(' ')).lower() + ng.gen_year(1900, 2020) + ng.gen_birthday()
                        name_dict[mid] = {'name': name, 'screen_name': screen_name}
                        s['entities']['user_mentions'][i]['name'] = name
                        s['entities']['user_mentions'][i]['screen_name'] = screen_name
                    else:
                        s['entities']['user_mentions'][i]['name'] = name_dict[mid]['name']
                        s['entities']['user_mentions'][i]['screen_name'] = name_dict[mid]['screen_name']
                if 'quoted_status' in s.keys():
                    qid = s['quoted_status']['user']['id']
                    if qid not in name_dict.keys():
                        name = ng.gen_two_words(split=' ', lowercase=False)
                        screen_name = ''.join(name.split(' ')).lower() + ng.gen_year(1900, 2020) + ng.gen_birthday()
                        name_dict[qid] = {'name': name, 'screen_name': screen_name}
                        s['quoted_status']['user']['name'] = name
                        s['quoted_status']['user']['screen_name'] = screen_name
                    else:
                        s['quoted_status']['user']['name'] = name_dict[qid]['name']
                        s['quoted_status']['user']['screen_name'] = name_dict[qid]['screen_name']
                    for i in range(len(s['quoted_status']['entities']['user_mentions'])):
                        mid = s['quoted_status']['entities']['user_mentions'][i]['id']
                        if mid not in name_dict.keys():
                            name = ng.gen_two_words(split=' ', lowercase=False)
                            screen_name = ''.join(name.split(' ')).lower() + ng.gen_year(1900, 2020) + ng.gen_birthday()
                            name_dict[mid] = {'name': name, 'screen_name': screen_name}
                            s['quoted_status']['entities']['user_mentions'][i]['name'] = name
                            s['quoted_status']['entities']['user_mentions'][i]['screen_name'] = screen_name
                        else:
                            s['quoted_status']['entities']['user_mentions'][i]['name'] = name_dict[mid]['name']
                            s['quoted_status']['entities']['user_mentions'][i]['screen_name'] = name_dict[mid]['screen_name']
                    if 'extended_tweet' in s['quoted_status'].keys():
                        for i in range(len(s['quoted_status']['extended_tweet']['entities']['user_mentions'])):
                            mid = s['quoted_status']['extended_tweet']['entities']['user_mentions'][i]['id']
                            if mid not in name_dict.keys():
                                name = ng.gen_two_words(split=' ', lowercase=False)
                                screen_name = ''.join(name.split(' ')).lower() + ng.gen_year(1900, 2020) + ng.gen_birthday()
                                name_dict[mid] = {'name': name, 'screen_name': screen_name}
                                s['quoted_status']['extended_tweet']['entities']['user_mentions'][i]['name'] = name
                                s['quoted_status']['extended_tweet']['entities']['user_mentions'][i]['screen_name'] = screen_name
                            else:
                                s['quoted_status']['extended_tweet']['entities']['user_mentions'][i]['name'] = name_dict[mid]['name']
                                s['quoted_status']['extended_tweet']['entities']['user_mentions'][i]['screen_name'] = name_dict[mid]['screen_name']
                line = json.dumps(s)
                fout.write(line + '\n')
            print('Processed records:', count)
    print('Generate time: {} seconds'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
