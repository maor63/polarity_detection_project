from functools import partial
from multiprocessing import freeze_support
from pathlib import Path
import json
import os

from polarity_quntification.polarization_algorithms.measuring_controversy_in_social_networks_through_nlp import \
    semantic_distance_pol
from polarity_quntification.polarization_algorithms.topic_propagation_ver2 import topic_propagation_pol
from polarity_quntification.polarization_algorithms.vocabulary_based_method_for_quantifying_controversy import vmqc_pol
from run_package import read_tweets_json, run_contrevercy_method_experiment_reddit, compute_polarization, \
    run_contrevercy_method_experiment

os.environ["METIS_DLL"] = "polarity_quntification/dll/metis.dll"
from polarity_quntification.polarization_algorithms.topic_propagation import topic_propagation, read_tweets_from_file
import polarity_quntification.polarization_algorithms.polarization_algorithms as pol
import networkx as nx
import polarity_quntification.partition_algorithms.partition_algorithms as pa
import networkx.algorithms.community as nx_comm
import pandas as pd
from tqdm.auto import tqdm


def run_reddit_structural_polarity_measures(reddit_path, output_path):
    cols = ['rwc_metis', 'arwc_metis', 'ebc_metis', 'gmck_metis', 'mblb_metis', 'mod_metis', 'ei_metis',
            'extei_metis', 'cond_metis', 'rwc_rsc', 'arwc_rsc', 'ebc_rsc', 'gmck_rsc', 'mblb_rsc', 'mod_rsc', 'ei_rsc',
            'extei_rsc', 'cond_rsc', 'size', 'ave_deg']
    run_contrevercy_method_experiment_reddit(reddit_path, output_path, cols, compute_polarization, reddit_graph_read_fn)


def run_reddit_topic_propagation(reddit_path, output_path, directed=True, topic_min_prob=0.0, topic_method='lda'
                                 , community_method='louvain', community_size_thresh=0.1):
    base_score_names = ['NMI', 'AMI', 'ARI']
    topic_propagation_pol_fn = partial(topic_propagation_pol, directed=directed, topic_min_prob=topic_min_prob,
                                       topic_method=topic_method, community_method=community_method,
                                       community_size_thresh=community_size_thresh)
    run_contrevercy_method_experiment_reddit(reddit_path, output_path, base_score_names, topic_propagation_pol_fn,
                                             reddit_graph_read_fn)


if __name__ == "__main__":
    freeze_support()
    dataset_type = 'kiran'
    export_results = True
    filter_retweets = True
    if dataset_type == 'kiran':
        network_path = Path('data/networks/retweet_networks/')
        suffix = '_threshold_largest_CC.txt'
        prefix = 'retweet_graph_'
        graph_read_fn = partial(nx.read_weighted_edgelist, delimiter=',', create_using=nx.DiGraph)
        read_tweet_fn = partial(read_tweets_from_file, filter_retweets=filter_retweets)
        tweets_file_name_suffix = '.txt'
    elif dataset_type == 'juan':
        network_path = Path('data/juan_gml/')
        suffix = '_r.gml'
        prefix = ''
        read_tweet_fn = read_tweets_json
        tweets_file_name_suffix = '_tweets.json'
        graph_read_fn = partial(nx.read_gml, label='name')

    # reddit_suffix = 'MadeMeSmile_01-03-2022_23-03-2022'
    # reddit_path = Path('data/') / reddit_suffix
    # reddit_graph_read_fn = nx.read_gml
    # output_base_path = Path('output/experiments')
    #
    # output_path = output_base_path / f'structural_polarity_quantification_scores_reddit_{reddit_suffix}.csv'
    # run_reddit_structural_polarity_measures(reddit_path, output_path)
    #
    # directed = True
    # topic_min_prob = 0.0
    # topic_method = 'lda'
    # community_method = 'louvain'
    # community_size_thresh = 0.1
    # net_type = 'directed:' if directed else 'undirected'
    # output_path = output_base_path / f'topic_propagation_{reddit_suffix}-{net_type}-{topic_method}-{topic_min_prob}-{community_method}-{community_size_thresh}.csv'
    # run_reddit_topic_propagation(reddit_path, output_path, directed=directed, topic_min_prob=topic_min_prob,
    #                              topic_method=topic_method, community_method=community_method,
    #                              community_size_thresh=community_size_thresh)

    ### twitter datasets ####
    for dataset_type in ['kiran', 'juan']:
        export_results = True
        filter_retweets = True
        if dataset_type == 'kiran':
            network_path = Path('data/networks/retweet_networks/')
            suffix = '_threshold_largest_CC.txt'
            prefix = 'retweet_graph_'
            graph_read_fn = partial(nx.read_weighted_edgelist, delimiter=',', create_using=nx.DiGraph)
            read_tweet_fn = partial(read_tweets_from_file, filter_retweets=filter_retweets)
            tweets_file_name_suffix = '.txt'
        elif dataset_type == 'juan':
            network_path = Path('data/juan_gml/')
            suffix = '_r.gml'
            prefix = ''
            read_tweet_fn = read_tweets_json
            tweets_file_name_suffix = '_tweets.json'
            graph_read_fn = partial(nx.read_gml, label='name')

        output_base_path = Path(f'output/experiments/{dataset_type}_dataset')
        if not output_base_path.exists():
            os.makedirs(output_base_path)

        tweet_path = Path('data/full_tweets/')
        output_path = output_base_path / f'VMQC_method_scores_{dataset_type}_dataset{"" if filter_retweets else "_with_retweets"}.csv'
        run_contrevercy_method_experiment(network_path, tweet_path, output_path, 'VMQC_score', vmqc_pol)

        output_path = output_base_path/ f'semantic_distance_scores_{dataset_type}_dataset{"" if filter_retweets else "_with_retweets"}.csv'
        run_contrevercy_method_experiment(network_path, tweet_path, output_path, 'semantic_distance', semantic_distance_pol)

        output_path = output_base_path / f'structural_scores_{dataset_type}_dataset.csv'
        cols = ['rwc_metis', 'arwc_metis', 'ebc_metis', 'gmck_metis', 'mblb_metis', 'mod_metis', 'ei_metis',
                'extei_metis', 'cond_metis', 'rwc_rsc', 'arwc_rsc', 'ebc_rsc', 'gmck_rsc', 'mblb_rsc', 'mod_rsc', 'ei_rsc',
                'extei_rsc', 'cond_rsc', 'size', 'ave_deg']
        run_contrevercy_method_experiment(network_path, tweet_path, output_path, cols, semantic_distance_pol)

        directed = True
        topic_min_prob = 0.0
        topic_method = 'lda'
        community_method = 'louvain'
        community_size_thresh = 0.1
        net_type = 'directed:' if directed else 'undirected'
        output_path = output_base_path / f'topic_propagation-{net_type}-{topic_method}-{topic_min_prob}-{community_method}-{community_size_thresh}.csv'
        # run_contrevercy_method_experiment(network_path, tweet_path, output_path,
        #                                                                     ['undirected_NMI', 'undirected_AMI', 'undirected_ARI'], undirected_topic_propagation_pol)
        #
        # run_reddit_topic_propagation(reddit_path, output_path, directed=directed, topic_min_prob=topic_min_prob,
        #                              topic_method=topic_method, community_method=community_method,
        #                              community_size_thresh=community_size_thresh)