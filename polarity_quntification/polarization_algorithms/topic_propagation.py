import os
from pathlib import Path

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore
from tqdm.auto import tqdm
import json
import re
from collections import Counter

import community as community_louvain
import emoji
import networkx as nx
import nltk
import pandas as pd
from bertopic import BERTopic
from gensim.matutils import *
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from tqdm.auto import tqdm

nltk.download('words')
nltk.download('stopwords')
from nltk.stem import PorterStemmer
import random
from karateclub.estimator import Estimator
from collections import defaultdict
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
from polarity_quntification.partition_algorithms.partition_algorithms import partition_spectral, partition_metis


class LabelPropagation(Estimator):
    r"""An implementation of `"Label Propagation Clustering" <https://arxiv.org/abs/0709.2938>`_
    from the Physical Review '07 paper "Near Linear Time Algorithm to Detect Community Structures
    in Large-Scale Networks". The tool executes a series of label propagations with unique labels.
    The final labels are used as cluster memberships.

    Args:
        seed (int): Random seed. Default is 42.
        iterations (int): Propagation iterations. Default is 100.
    """

    def __init__(self, seed: int = 42, iterations=100, loudest_neighbor=False):
        self.seed = seed
        self.iterations = iterations
        self._loudest_neighbor = loudest_neighbor
        self._directed = True

    def _make_a_pick(self, neighbors, weighted):
        """
        Choosing a neighbor from a propagation source node.

        Arg types:
            * **neigbours** *(list)* - Neighbouring nodes.
        """
        scores = defaultdict(int)
        if self._loudest_neighbor:
            scores = dict(neighbors)
            top = [self._labels[neighbor] for neighbor, val in scores.items() if val == max(scores.values())]
        else:
            for neighbor, weight in neighbors:
                neighbor_label = self._labels[neighbor]
                scores[neighbor_label] += weight if weighted else 1
            top = [key for key, val in scores.items() if val == max(scores.values())]
        return random.sample(top, 1)[0]

    def _get_neighbors(self, node, in_edges=True):
        if self._directed:
            if in_edges:
                edges_gen = self._graph.in_edges(node, data=True)
            else:
                edges_gen = self._graph.out_edges(node, data=True)
        else:
            edges_gen = self._graph.edges(node, data=True)
        return [(neb, data.get('weight', 1)) for src, neb, data in edges_gen if self._labels[neb] >= 0]

    def _do_a_propagation(self, weighted, in_edges=True):
        """
        Doing a propagation round.
        """
        random.shuffle(self._nodes)
        new_labels = {}
        for node in self._nodes:
            neighbors = self._get_neighbors(node, in_edges=in_edges)
            if len(neighbors) > 0:
                pick = self._make_a_pick(neighbors, weighted)
            else:
                pick = self._labels[node]
            new_labels[node] = pick
        error = 0
        for user in self._labels:
            if self._labels[user] != new_labels[user]:
                error += 1

        self._labels = new_labels
        self._error = error

    def fit(self, graph, init_labels=None, to_undirected=True, weighted=True, in_edges=True):
        """
        Fitting a Label Propagation clustering model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self._set_seed()
        #         graph = self._check_graph(graph)
        if to_undirected:
            self._graph = graph.to_undirected()
            self._directed = False
        else:
            self._graph = graph.copy()

        self._nodes = [node for node in self._graph.nodes()]
        if init_labels:
            self._labels = dict(init_labels)
            # for node in self._graph.nodes():
            #     if node not in self._labels:
            #         self._labels[node] = -1
        else:
            self._labels = {node: i for i, node in enumerate(self._graph.nodes())}
        random.seed(self.seed)
        self._error = len(self._nodes)
        for _ in tqdm(range(self.iterations), desc='label propagation iteration'):
            self._do_a_propagation(weighted, in_edges=in_edges)
            if self._error < 0.01 * len(self._nodes):
                break
        print('error', self._error)

    def get_memberships(self):
        r"""Getting the cluster membership of nodes.

        Return types:
            * **memberships** *(dict)* - Node cluster memberships.
        """
        memberships = self._labels
        return memberships


def read_tweets_from_file(file_path, filter_retweets=False):
    input_tweets = []
    with open(file_path, encoding='utf-8') as f:
        if filter_retweets:
            for line in tqdm(f, desc='read tweets from file'):
                t = json.loads(line)
                if not t['text'].startswith('RT @'):
                    input_tweets.append(t)
        else:
            for line in tqdm(f, desc='read tweets and retweets from file'):
                input_tweets.append(json.loads(line))
    return input_tweets


def cleaner(tweet, stemmer, tknzr, topic_words, stem=False):
    tweet = re.sub("@[A-Za-z0-9]+", "", tweet)  # Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet)  # Remove http links
    tweet = " ".join(tweet.split())
    tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI)  # Remove Emojis
    tweet = tweet.replace("#", "").replace("_", " ").replace('n ’ t', 'n’t')  # Remove hashtag sign but keep the text
    tweet_words = [w for w in tknzr.tokenize(tweet) if w.lower() not in STOPWORDS | topic_words]
    if stem:
        tweet_words = [stemmer.stem(w) for w in tweet_words]
    tweet = " ".join(tweet_words)
    return tweet.lower()


def remove_short_tokens(tokens, limit=2):
    return [t.lower() for t in tokens if len(t) >= limit]


def export_topic_tweet_csv(graph_tweets, min_words, output_path, probs, rows, seen_tweets, topics):
    for tweet, topic, prob in tqdm(zip(graph_tweets, topics, probs), desc='topic to tweets', total=len(graph_tweets)):
        username = tweet['user']['screen_name']
        if len(tweet['text'].split()) >= min_words and tweet['id'] not in seen_tweets:
            link = f'https://twitter.com/{username}/status/{tweet["id_str"]}'
            rows.append([topic, prob, -1, tweet['text'], link, username])
        seen_tweets.add(tweet['id'])
    pd.DataFrame(rows, columns=['topic', 'topic_prob', 'community', 'text', 'link', 'user']).to_csv(
        output_path / f'{network_name}_tweets_with_more_{min_words}_words_topics{num_topics}_0.55.csv')


def mark_user_by_topic(graph_nodes, graph_tweets, probs, topics, topic_treshold=0.55):
    user_to_topic = defaultdict(list)

    for tweet, topic, prob in tqdm(zip(graph_tweets, topics, probs), desc='calc user topics', total=len(graph_tweets)):
        username = tweet['user']['screen_name']
        user_to_topic[username].append(topic if prob > topic_treshold else -1)
    topic_partition = {t['user']['screen_name']: Counter(user_to_topic[t['user']['screen_name']]).most_common(1)[0][0]
                       for t in tqdm(graph_tweets, desc='topic partition')}
    missing = 0
    for u in graph_nodes:
        if u not in topic_partition:
            topic_partition[u] = -1
            missing +=1
    print('users without init topic', missing)
    return topic_partition


def load_interaction_graph(network_path, network_name):
    edges_df = pd.read_csv(network_path / f'retweet_graph_{network_name}_threshold_largest_CC.txt', header=None,
                           names=['src', 'dst', 'w'])
    weighted_edges = [v for v in edges_df.itertuples(index=None, name=None)]
    G = nx.DiGraph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
    G.add_weighted_edges_from(weighted_edges)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def topic_community_correlation(label_popagation_partition, topic_diffusion):
    community_vals = []
    topic_vals = []
    for user, topic in topic_diffusion.items():
        if topic >= 0:
            community_vals.append(label_popagation_partition[user])
            topic_vals.append(topic)
    directed_nmi = normalized_mutual_info_score(community_vals, topic_vals)
    directed_ami = adjusted_mutual_info_score(community_vals, topic_vals)
    directed_ari = adjusted_rand_score(community_vals, topic_vals)
    return directed_nmi, directed_ami, directed_ari


def topic_propagation(network_name, G, tweets, network_type='retweet'):
    # Read Tweets
    # tweets = read_tweets_from_file(tweet_path / f'{network_name}.txt', filter_retweets=False)

    stemmer = PorterStemmer()
    tknzr = TweetTokenizer()
    topic_words = {network_name}

    graph_nodes = set(G.nodes())
    graph_tweets = [t for t in tqdm(tweets, desc='get tweets filtered by graph') if
                    t['user']['screen_name'] in graph_nodes]

    # Bert topic modeling
    cleaner_args = {
        'stemmer': stemmer,
        'tknzr': tknzr,
        'topic_words': topic_words,
        'stem': True,
    }

    ##### BERTopic ###########
    # num_topics = 'bert'
    # tweet_texts = [cleaner(t['text'], **cleaner_args) for t in tqdm(graph_tweets, desc='clear tweets')]
    # topic_model = BERTopic(verbose=True)
    # topics, probs = topic_model.fit_transform(tweet_texts)

    # Mark user by topic
    # topic_treshold = 0.55
    # topic_partition = mark_user_by_topic(graph_nodes, graph_tweets, probs, topics, topic_treshold=topic_treshold)
    # probs = np.ones(len(topics)) if probs is None else probs

    ######## LDA #############
    num_topics = 10
    tweet_tokens = [remove_short_tokens(tknzr.tokenize(cleaner(t['text'], **cleaner_args)), limit=4) for t in
                    tqdm(graph_tweets, desc='prepare tweets')]
    text_dict = Dictionary(tweet_tokens)
    tweets_bow = [text_dict.doc2bow(tweet) for tweet in tweet_tokens]

    lda_path = Path('data/lda_models')
    if not lda_path.exists():
        os.makedirs(lda_path)

    lda_model_name = lda_path / f'lda{num_topics}-{network_name}.model'
    if lda_model_name.exists():
        lda_model = LdaModel.load(str(lda_model_name))
    else:
        lda_model = LdaMulticore(corpus=tweets_bow,
                                 id2word=text_dict,
                                 num_topics=num_topics,
                                 random_state=1,
                                 passes=10,
                                 workers=3)

        lda_model.save(str(lda_model_name))
    topic_treshold = 0.3
    tweet_topic_probs = lda_model.get_document_topics(tweets_bow)

    # Mark user by topic
    user_to_topic = defaultdict(list)
    tweets_users = [t['user']['screen_name'] for t in graph_tweets]
    n_users = len(tweets_users)
    probs = []
    topics = []
    for username, tweet_topic_prob in tqdm(zip(tweets_users, tweet_topic_probs), desc='calc user topics', total=n_users):
        max_topic, prob = max(tweet_topic_prob, key=lambda x: x[1])
        probs.append(prob)
        topics.append(max_topic)
        # user_to_topic[username].append(max_topic if prob > topic_treshold else -1)

    # topic_partition = {u: Counter(user_to_topic[u]).most_common(1)[0][0] for u in user_to_topic}
    topic_partition = mark_user_by_topic(graph_nodes, graph_tweets, probs, topics, topic_treshold=topic_treshold)

    graph_name = f'{network_name}_{network_type}_topics{num_topics}_fixed_topic_propagation_{topic_treshold}'

    G_copy = G.copy(as_view=False)
    nx.set_node_attributes(G_copy, topic_partition, "group")
    # node_topics = [topic_partition[n] for n in G_copy.nodes()]

    iterations = 100
    for weighted in [True]:
        for directed, in_edges in [(False, True), (True, False)]:
            graph_type = f'{"directed" if directed else "undirected"}'
            if directed:
                graph_type += f'{"_in_edges" if in_edges else "_out_edges"}'
            graph_type += f'{"_weighted" if weighted else ""}'
            lp_model = LabelPropagation(iterations=iterations, loudest_neighbor=False)
            print('Topic propagation', graph_type)
            lp_model.fit(G_copy, init_labels=topic_partition, to_undirected=not directed, weighted=weighted,
                         in_edges=in_edges)
            label_popagation_partition = lp_model.get_memberships()
            nx.set_node_attributes(G_copy, label_popagation_partition, graph_type)
            #         nx.set_node_attributes(G_copy, {k: str(v) for k, v in label_popagation_partition.items()}, "title")
            print()

    graph_name = f'{graph_name}_community_louvain'
    louvain_partition = community_louvain.best_partition(G_copy.to_undirected())
    nx.set_node_attributes(G_copy, louvain_partition, 'louvain_partition')
    # nx.write_gexf(G_copy, f"{graph_name}.gexf")

    # extract discourse topic data
    # rows = []
    # min_words = 3
    # seen_tweets = set()
    # export_topic_tweet_csv(graph_tweets, min_words, output_path, probs, rows, seen_tweets, topics)
    print('-------------- louvain_partition ------------------')
    topic_diffusion = nx.get_node_attributes(G_copy, 'directed_out_edges_weighted')
    directed_nmi, directed_ami, directed_ari = topic_community_correlation(louvain_partition, topic_diffusion)
    print('directed_NMI:', directed_nmi)
    print('directed_AMI:', directed_ami)
    print('directed_ARI:', directed_ari)

    topic_diffusion = nx.get_node_attributes(G_copy, 'undirected_weighted')
    undirected_nmi, undirected_ami, undirected_ari = topic_community_correlation(louvain_partition, topic_diffusion)
    print('undirected_NMI:', undirected_nmi)
    print('undirected_AMI:', undirected_ami)
    print('undirected_ARI:', undirected_ari)
    louvain_scores = [directed_nmi, directed_ami, directed_ari, undirected_nmi, undirected_ami, undirected_ari]

    # print('-------------- metis_partition ------------------')
    # metis_partition = partition_metis(G)
    # nx.set_node_attributes(G_copy, metis_partition, 'metis_partition')
    #
    # topic_diffusion = nx.get_node_attributes(G_copy, 'directed_out_edges_weighted')
    # directed_nmi, directed_ami, directed_ari = topic_community_correlation(metis_partition, topic_diffusion)
    # print('directed_NMI:', directed_nmi)
    # print('directed_AMI:', directed_ami)
    # print('directed_ARI:', directed_ari)
    #
    # topic_diffusion = nx.get_node_attributes(G_copy, 'undirected_weighted')
    # undirected_nmi, undirected_ami, undirected_ari = topic_community_correlation(metis_partition, topic_diffusion)
    # print('undirected_NMI:', undirected_nmi)
    # print('undirected_AMI:', undirected_ami)
    # print('undirected_ARI:', undirected_ari)
    # metis_scores = [directed_nmi, directed_ami, directed_ari, undirected_nmi, undirected_ami, undirected_ari]
    #
    # print('-------------- rsc_partition ------------------')
    # rsc_partition = partition_spectral(G)
    # nx.set_node_attributes(G_copy, rsc_partition, 'rsc_partition')
    #
    # topic_diffusion = nx.get_node_attributes(G_copy, 'directed_out_edges_weighted')
    # directed_nmi, directed_ami, directed_ari = topic_community_correlation(rsc_partition, topic_diffusion)
    # print('directed_NMI:', directed_nmi)
    # print('directed_AMI:', directed_ami)
    # print('directed_ARI:', directed_ari)
    #
    # topic_diffusion = nx.get_node_attributes(G_copy, 'undirected_weighted')
    # undirected_nmi, undirected_ami, undirected_ari = topic_community_correlation(rsc_partition, topic_diffusion)
    # print('undirected_NMI:', undirected_nmi)
    # print('undirected_AMI:', undirected_ami)
    # print('undirected_ARI:', undirected_ari)
    # rsc_scores = [directed_nmi, directed_ami, directed_ari, undirected_nmi, undirected_ami, undirected_ari]

    nx.write_gexf(G_copy, f"{graph_name}.gexf")

    # return louvain_scores, metis_scores, rsc_scores
    return louvain_scores


if __name__ == "__main__":
    topic_propagation()
