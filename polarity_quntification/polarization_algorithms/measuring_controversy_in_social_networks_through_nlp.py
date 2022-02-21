'''
This is a Python implementation of the method presented in "Measuring Controversy in Social Networks
through NLP" https://www.cs.ucf.edu/spire2020/wp-content/uploads/2020/10/Measuring-Controversy-in-Social-Networks-through-NLP-PRESENTATION.pdf.
The implementation is based on the original R code https://github.com/jmanuoz/Measuring-controversy-in-Social-Networks-through-NLP.
'''
import json
import os
from collections import Counter
from functools import partial
from operator import itemgetter
from pathlib import Path

import community as community_louvain
import fasttext
import networkx as nx
import numpy as np
from scipy.spatial import distance

from polarity_quntification.polarization_algorithms.vocabulary_based_method_for_quantifying_controversy import \
    prepare_train_file, create_user_text_df


def semantic_distance_pol(G, tweets, get_tweet_username, verbos=False, graph_name='none'):
    G_copy = G.copy()
    output_path = Path('output/semantic_distance_method/')
    if not output_path.exists():
        os.makedirs(output_path)

    louvain_partition = community_louvain.best_partition(G_copy.to_undirected())
    largest_communities = Counter(louvain_partition.values()).most_common(2)
    community1 = largest_communities[0][0]
    community2 = largest_communities[1][0]
    user_text = create_user_text_df(get_tweet_username, louvain_partition, tweets)

    train_file_path = str(output_path / f'{graph_name}-train.txt')
    community1_user_text, community2_user_text = prepare_train_file(community1, community2, train_file_path, user_text)

    node_count = len(G_copy.nodes())
    central_node_count = int(node_count * 0.3)
    hubs, authorities = nx.hits(G_copy, normalized=True)
    hubs = sorted(hubs, reverse=True, key=itemgetter(1))[:central_node_count]
    authorities = sorted(authorities, reverse=True, key=itemgetter(1))[:central_node_count]
    hubs, authorities = set(hubs), set(authorities)
    central_users = hubs | authorities
    community1_central_texts = community1_user_text[community1_user_text['username'].isin(central_users)]['text']
    # community1_central_texts.to_csv(output_path / f'{graph_name}-C1.txt', index=False, header=None)
    community2_central_texts = community2_user_text[community2_user_text['username'].isin(central_users)]['text']
    # community2_central_texts.to_csv(output_path / f'{graph_name}-C2.txt', index=False, header=None)

    fasttext_model = fasttext.train_supervised(train_file_path, epoch=20, dim=200, wordNgrams=2, ws=5, seed=123)
    vecs1 = [fasttext_model.get_sentence_vector(txt) for txt in community1_central_texts]
    vecs2 = [fasttext_model.get_sentence_vector(txt) for txt in community2_central_texts]

    c1 = np.vstack(vecs1)
    c2 = np.vstack(vecs2)
    X = np.concatenate((c1, c2), axis=0)
    # divide 2 clusters

    # calculate centroids
    cent1 = c1.mean(axis=0)
    cent2 = c2.mean(axis=0)
    cent0 = X.mean(axis=0)

    v = np.cov(X.T)
    SS0 = 0
    for row in X:
        # SS0 = SS0 + distance.mahalanobis(row,cent0,v)
        SS0 = SS0 + distance.cosine(row, cent0)
        # SS0 = SS0 + distance.euclidean(row,cent0)
        # SS0 = SS0 + distance.cityblock(row,cent0)
    v = np.cov(c1.T)
    SS1 = 0
    for row in c1:
        # SS1 = SS1 + distance.mahalanobis(row,cent1,v)
        SS1 = SS1 + distance.cosine(row, cent1)
        # SS1 = SS1 + distance.euclidean(row,cent1)
        # SS1 = SS1 + distance.cityblock(row,cent1)
    v = np.cov(c2.T)
    SS2 = 0
    for row in c2:
        # SS2 = SS2 + distance.mahalanobis(row,cent2,v)
        SS2 = SS2 + distance.cosine(row, cent2)
        # SS2 = SS2 + distance.euclidean(row,cent2)
        # SS2 = SS2 + distance.cityblock(row,cent2)
    # Controversy index
    controversy_score = (SS1 + SS2) / SS0
    print(controversy_score)
    return controversy_score


def read_tweet_fn(f_file):
    with open(f_file, encoding='latin-1') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    # R package names
    import rpy2.robjects.packages as rpackages

    # import R's utility package
    utils = rpackages.importr('utils')
    packnames = ('plyr', 'textclean')

    # R vector of strings
    from rpy2.robjects.vectors import StrVector

    # Selectively install what needs to be install.
    # We are fancy, just because we can.
    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))

    tweets_path = 'data/full_tweets/area51_tweets.json'
    graph_path = Path('data/juan_gml/area51_r.gml')
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

    semantic_distance_pol(G, tweets, get_tweet_username, verbos=False, graph_name=graph_name)
