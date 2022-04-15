'''
This is a Python implementation of the method presented in "Vocabulary-based Method for Quantifying
Controversy in Social Media" https://arxiv.org/pdf/2001.09899.pdf.
The implementation is based on the original R code https://github.com/jmanuoz/Vocabulary-based-Method-for-Quantify-Controversy
'''
import json
import time
from functools import partial
import random
from cdlib import algorithms
import fasttext
import community as community_louvain
from collections import Counter
import pandas as pd
from nltk import TweetTokenizer
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import networkx as nx
import os
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from tqdm.auto import tqdm
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore

from polarity_quntification.polarization_algorithms.topic_propagation import topic_community_correlation
from polarity_quntification.polarization_algorithms.vocabulary_based_method_for_quantifying_controversy import \
    create_user_text_df, clean_text_r_code


def topic_propagation_pol(G, tweets, get_tweet_username, verbos=False, graph_name='none', directed=True,
                          topic_min_prob=0.0, topic_method='lda', community_method='louvain', community_size_thresh=0.1):
    G_copy = G.copy()
    if not directed:
        G_copy = G_copy.to_undirected()
    output_path = Path('output/topic_propagation_method/')
    if not output_path.exists():
        os.makedirs(output_path)

    if community_method == 'louvain':
        community_partition = community_louvain.best_partition(G_copy.to_undirected())
    elif community_method == 'leiden':
        community_partition = algorithms.leiden(G_copy.to_undirected())
        community_partition = {k: vs[0] for k, vs in community_partition.to_node_community_map().items()}
    N = len(G_copy.nodes())
    largest_communities = [c for c, size in Counter(community_partition.values()).items() if size > N * community_size_thresh]
    print('number of large communities:', len(largest_communities))
    user_text = create_user_text_df(get_tweet_username, community_partition, tweets)

    user_text['text_clean'] = clean_text_r_code(user_text['text'])

    tknzr = TweetTokenizer()
    tweet_tokens = [tknzr.tokenize(t) for t in user_text['text_clean']]
    text_dict = Dictionary(tweet_tokens)
    tweets_bow = [text_dict.doc2bow(tweet) for tweet in tweet_tokens]

    num_topics = max(len(largest_communities), 2)
    if topic_method == 'lda':
        lda_path = output_path / 'lda_models/'
        if not lda_path.exists():
            os.makedirs(lda_path)
        lda_model_name = lda_path / f'lda{num_topics}-{graph_name}.model'

        if lda_model_name.exists():
            lda_model = LdaModel.load(str(lda_model_name))
        else:
            lda_model = LdaMulticore(corpus=tweets_bow,
                                     id2word=text_dict,
                                     num_topics=num_topics,
                                     random_state=1,
                                     passes=10,
                                     workers=3)

            # lda_model.save(str(lda_model_name))

        tweet_topic_probs = lda_model.get_document_topics(tweets_bow)

    topic_prob_rows = [max(tweet_topic_prob, key=lambda x: x[1]) for tweet_topic_prob in tweet_topic_probs]
    topic_prob_df = pd.DataFrame(topic_prob_rows, columns=['topic', 'prob'])
    topic_prob_df[topic_prob_df['prob'] < topic_min_prob].loc[:, 'topic'] = -1

    user_to_topic = dict(zip(user_text['username'], topic_prob_df['topic']))
    topic_diffusion = label_propagation(G_copy, user_to_topic, tol=10 ** -5)

    directed_nmi, directed_ami, directed_ari = topic_community_correlation(community_partition, topic_diffusion)
    # net_type = 'directed:' if directed else 'undirected'
    return [directed_nmi, directed_ami, directed_ari]


def label_propagation(G, v_current, tol=10 ** -5):
    for n in G.nodes():
        if n not in v_current:
            v_current[n] = -1
    notconverged = len(v_current)
    times = 0
    Aij = nx.adjacency_matrix(G)
    n_nodes = len(G.nodes())
    all_nodes = list(G.nodes())
    v_new = {}

    # Do as many times as required for convergence
    for i in tqdm(range(min(len(v_current), 300)), desc='label propagation'):
        if notconverged == 0:
            break

        random.shuffle(all_nodes)
        v_current_array = []
        v_new_array = []
        for node in all_nodes:
            edges_gen = G.edges(node, data=True)
            neighbor_topic_count = Counter()
            for src, neb, data in edges_gen:
                if v_current[neb] >= 0:
                    neighbor_topic_count[v_current[neb]] += data.get('weight', 1)
            if len(neighbor_topic_count) > 0:
                v_new[node] = neighbor_topic_count.most_common(1)[0][0]
            else:
                v_new[node] = v_current[node]

            v_current_array.append(v_current[node])
            v_new_array.append(v_new[node])
        diff = np.array(v_current_array) != np.array(v_new_array)
        notconverged = diff.sum()
        v_current = v_new.copy()
        v_new = {}
    print('\n not converged', notconverged)
    return v_current


def read_tweet_fn(f_file):
    with open(f_file, encoding='latin-1') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    graph_name = 'bolsonaro27'
    tweets_path = 'data/full_tweets/%s_tweets.json' % graph_name
    graph_path = Path('data/juan_gml/%s_r.gml' % graph_name)
    suffix = '_r.gml'
    prefix = ''

    tweets_file_name_suffix = '_tweets.json'
    graph_read_fn = partial(nx.read_gml, label='name')
    graph_name = graph_path.name.replace(suffix, '').replace(prefix, '')

    G = graph_read_fn(graph_path)
    G.graph['edge_weight_attr'] = 'weight'
    edge_weight = nx.get_edge_attributes(G, 'weight')
    nx.set_edge_attributes(G, {k: int(v) for k, v in edge_weight.items()}, 'weight')
    tweets = read_tweet_fn(tweets_path)

    if len(tweets) > 0 and 'user' in tweets[0]:
        get_tweet_username = lambda t: t['user']['screen_name']
    else:
        get_tweet_username = lambda t: t['screen_name']

    topic_propagation_pol(G, tweets, get_tweet_username, verbos=False, graph_name=graph_name)
