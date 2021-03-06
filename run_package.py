import json
import os
from functools import partial
from multiprocessing import freeze_support
from pathlib import Path

from polarity_quntification.polarization_algorithms.measuring_controversy_in_social_networks_through_nlp import \
    semantic_distance_pol
from polarity_quntification.polarization_algorithms.topic_propagation_ver2 import topic_propagation_pol
from polarity_quntification.polarization_algorithms.vocabulary_based_method_for_quantifying_controversy import vmqc_pol

os.environ["METIS_DLL"] = "polarity_quntification/dll/metis.dll"

from polarity_quntification.polarization_algorithms.topic_propagation import topic_propagation, read_tweets_from_file
import polarity_quntification.polarization_algorithms.polarization_algorithms as pol
import networkx as nx
import polarity_quntification.partition_algorithms.partition_algorithms as pa
import networkx.algorithms.community as nx_comm
import pandas as pd
from tqdm.auto import tqdm


def get_giant_component(dG):
    Gcc = sorted(nx.connected_components(dG), key=len, reverse=True)
    G_Giant = dG.subgraph(Gcc[0])
    G = nx.convert_node_labels_to_integers(G_Giant)

    return G


def get_average_degree(dG):
    degree_sequence = [d for n, d in dG.degree()]
    average_degree = sum(degree_sequence) / len(degree_sequence)

    return average_degree


def compute_polarization(dG, *args, **kwargs):
    G = nx.to_undirected(dG)
    G = get_giant_component(G)

    ms_rsc = pa.partition_spectral(G)
    T_rsc = [node for node in ms_rsc if ms_rsc[node] == 0]
    S_rsc = [node for node in ms_rsc if ms_rsc[node] == 1]
    # print("RSC completed.")

    ms_metis = pa.partition_metis(G)
    T_metis = [node for node in ms_metis if ms_metis[node] == 0]
    S_metis = [node for node in ms_metis if ms_metis[node] == 1]
    # print("METIS completed.")

    n_sim, n_walks = 5, int(1e4)

    rwc_rsc = pol.random_walk_pol(G, ms_rsc, 10, n_sim, n_walks)
    rwc_metis = pol.random_walk_pol(G, ms_metis, 10, n_sim, n_walks)
    print("RWC nonadaptive completed.")

    arwc_rsc = pol.random_walk_pol(G, ms_rsc, 0.01, n_sim, n_walks)
    arwc_metis = pol.random_walk_pol(G, ms_metis, 0.01, n_sim, n_walks)
    print("RWC adaptive completed.")

    cond_rsc = 1 - nx.conductance(G, S_rsc, T_rsc)
    cond_metis = 1 - nx.conductance(G, S_metis, T_metis)
    print("Conductance completed.")

    mod_rsc = nx_comm.modularity(G, [S_rsc, T_rsc])
    mod_metis = nx_comm.modularity(G, [S_metis, T_metis])
    print("Modularity completed.")

    ei_rsc = -1 * pol.krackhardt_ratio_pol(G, ms_rsc)
    ei_metis = -1 * pol.krackhardt_ratio_pol(G, ms_metis)
    print("E-I completed.")

    extei_rsc = -1 * pol.extended_krackhardt_ratio_pol(G, ms_rsc)
    extei_metis = -1 * pol.extended_krackhardt_ratio_pol(G, ms_metis)
    print("Extended E-I completed.")

    # ebc_rsc = pol.betweenness_pol(G, ms_rsc)
    # ebc_metis = pol.betweenness_pol(G, ms_metis)
    ebc_rsc = -1
    ebc_metis = -1
    print("BCC completed.")

    gmck_rsc = pol.gmck_pol(G, ms_rsc)
    gmck_metis = pol.gmck_pol(G, ms_metis)
    print("GMCK completed.")

    mblb_rsc = pol.dipole_pol(G, ms_rsc)
    mblb_metis = pol.dipole_pol(G, ms_metis)
    print("MBLB completed.")

    ave_deg = get_average_degree(G)
    size = len(G)

    infopack = [rwc_metis, arwc_metis, ebc_metis, gmck_metis, mblb_metis, mod_metis, ei_metis, extei_metis, cond_metis,
                rwc_rsc, arwc_rsc, ebc_rsc, gmck_rsc, mblb_rsc, mod_rsc, ei_rsc, extei_rsc, cond_rsc,
                size, ave_deg]

    return infopack


def read_tweets_json(f_file):
    with open(f_file, encoding='latin-1') as f:
        data = json.load(f)
    return data


def run_contrevercy_method_experiment(network_path, tweet_path, output_path, score_name, semantic_polarity_score):
    if isinstance(score_name, list):
        cols = ['graph_name'] + score_name
    else:
        cols = ['graph_name', score_name]
    if output_path.exists():
        processed_graphs = pd.read_csv(output_path)['graph_name'].tolist()
    else:
        processed_graphs = []
    graphs_n = len(list(network_path.iterdir())) - len(processed_graphs)
    for edgelist_file in tqdm(network_path.iterdir(), desc='process graphs', total=graphs_n):
        if edgelist_file.name.endswith(suffix):
            graph_name = edgelist_file.name.replace(suffix, '').replace(prefix, '')
            if graph_name not in processed_graphs and (tweet_path / f'{graph_name}{tweets_file_name_suffix}').exists():
                print(graph_name)
                # G = nx.read_weighted_edgelist(edgelist_file, delimiter=',', create_using=nx.DiGraph)
                G = graph_read_fn(edgelist_file)

                G.graph['edge_weight_attr'] = 'weight'
                edge_weight = nx.get_edge_attributes(G, 'weight')
                nx.set_edge_attributes(G, {k: int(v) for k, v in edge_weight.items()}, 'weight')

                # tweets = read_tweets_from_file(tweet_path / f'{graph_name}.txt', filter_retweets=False)
                tweets = read_tweet_fn(tweet_path / f'{graph_name}{tweets_file_name_suffix}')

                if len(tweets) > 0 and 'user' in tweets[0]:
                    get_tweet_username = lambda t: t['user']['screen_name']
                else:
                    get_tweet_username = lambda t: t['screen_name']

                # louvain_scores , metis_scores , rsc_scores = topic_propagation(graph_name, G, tweets, network_type='retweet')
                vmqc_score = semantic_polarity_score(G, tweets, get_tweet_username, verbos=False, graph_name=graph_name, network_name=graph_name)
                if isinstance(vmqc_score, list):
                    row = [graph_name] + vmqc_score
                else:
                    row = [graph_name, vmqc_score]
                if output_path.exists():
                    pd.DataFrame([row], columns=cols).to_csv(output_path, index=False, header=False, mode='a')
                else:
                    pd.DataFrame([row], columns=cols).to_csv(output_path, index=False)


def run_contrevercy_method_experiment_reddit(submissions_path, output_path, score_name, semantic_polarity_score,
                                             graph_read_fn):
    if isinstance(score_name, list):
        cols = ['graph_name'] + score_name
    else:
        cols = ['graph_name', score_name]
    if output_path.exists():
        processed_graphs = pd.read_csv(output_path)['graph_name'].tolist()
    else:
        processed_graphs = []
    graphs_n = len(list((submissions_path / 'comments').iterdir())) - len(processed_graphs)
    for comments_dir in tqdm((submissions_path / 'comments').iterdir(), desc='process graphs', total=graphs_n):
        graph_name = comments_dir
        if graph_name not in processed_graphs:
            print(graph_name)
            # G = nx.read_weighted_edgelist(edgelist_file, delimiter=',', create_using=nx.DiGraph)
            G = graph_read_fn(comments_dir / 'interaction_graph.gml')
            G.graph['edge_weight_attr'] = 'weight'
            edge_weight = nx.get_edge_attributes(G, 'weight')
            nx.set_edge_attributes(G, {k: int(v) for k, v in edge_weight.items()}, 'weight')

            # tweets = read_tweets_from_file(tweet_path / f'{graph_name}.txt', filter_retweets=False)
            tweets = [t._asdict() for t in pd.read_csv(comments_dir / f'comments.csv').itertuples(index=False)]

            get_tweet_username = lambda t: t['author']

            # louvain_scores , metis_scores , rsc_scores = topic_propagation(graph_name, G, tweets, network_type='retweet')
            vmqc_score = semantic_polarity_score(G, tweets, get_tweet_username, network_name=graph_name, verbos=False, graph_name=graph_name, )
            if isinstance(vmqc_score, list):
                row = [graph_name] + vmqc_score
            else:
                row = [graph_name, vmqc_score]
            if output_path.exists():
                pd.DataFrame([row], columns=cols).to_csv(output_path, index=False, header=False, mode='a')
            else:
                pd.DataFrame([row], columns=cols).to_csv(output_path, index=False)


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

    ######### reddit dataset ################
    # reddit_suffix = 'MadeMeSmile_01-03-2022_23-03-2022'
    # reddit_path = Path('data/') / reddit_suffix
    # reddit_graph_read_fn = nx.read_gml
    #
    # output_path = Path(f'structural_polarity_quantification_scores_reddit_mademesmile_dataset.csv')
    # cols = ['rwc_metis', 'arwc_metis', 'ebc_metis', 'gmck_metis', 'mblb_metis', 'mod_metis', 'ei_metis',
    #         'extei_metis', 'cond_metis', 'rwc_rsc', 'arwc_rsc', 'ebc_rsc', 'gmck_rsc', 'mblb_rsc', 'mod_rsc', 'ei_rsc',
    #         'extei_rsc', 'cond_rsc', 'size', 'ave_deg']
    #
    # if output_path.exists():
    #     processed_graphs = pd.read_csv(output_path)['graph_name'].tolist()
    # else:
    #     processed_graphs = []
    # run_contrevercy_method_experiment_reddit(reddit_path, output_path, cols, compute_polarization, reddit_graph_read_fn)
    #
    # output_path = Path(f'topic_propagation_scores_reddit_mademesmile_dataset.csv')
    # base_score_names = ['directed_NMI', 'directed_AMI', 'directed_ARI', 'undirected_NMI', 'undirected_AMI',
    #                     'undirected_ARI']
    # run_contrevercy_method_experiment_reddit(reddit_path, output_path, base_score_names, topic_propagation,
    #                                          reddit_graph_read_fn)

    ########################### stractural methods #################################
    # output_path = Path(f'structural_polarity_quantification_scores_{dataset_type}_dataset.csv')
    # cols = ['graph_name', 'rwc_metis', 'arwc_metis', 'ebc_metis', 'gmck_metis', 'mblb_metis', 'mod_metis', 'ei_metis',
    #         'extei_metis', 'cond_metis', 'rwc_rsc', 'arwc_rsc', 'ebc_rsc', 'gmck_rsc', 'mblb_rsc', 'mod_rsc', 'ei_rsc',
    #         'extei_rsc', 'cond_rsc', 'size', 'ave_deg']
    #
    # if output_path.exists():
    #     processed_graphs = pd.read_csv(output_path)['graph_name'].tolist()
    # else:
    #     processed_graphs = []
    #
    # graphs_n = len(list(network_path.iterdir())) - len(processed_graphs)
    # for edgelist_file in tqdm(network_path.iterdir(), desc='process graphs', total=graphs_n):
    #     if edgelist_file.name.endswith(suffix):
    #         graph_name = edgelist_file.name.replace(suffix, '').replace(prefix, '')
    #         if graph_name not in processed_graphs:
    #             print(graph_name)
    #             # G = nx.read_weighted_edgelist(edgelist_file, delimiter=',')
    #             G = graph_read_fn(edgelist_file)
    #             G = G.to_undirected()
    #             G.graph['edge_weight_attr'] = 'weight'
    #             edge_weight = nx.get_edge_attributes(G, 'weight')
    #             nx.set_edge_attributes(G, {k: int(v) for k, v in edge_weight.items()}, 'weight')
    #
    #             row = [graph_name] + compute_polarization(G)
    #             if output_path.exists():
    #                 pd.DataFrame([row], columns=cols).to_csv(output_path, index=False, header=False, mode='a')
    #             else:
    #                 pd.DataFrame([row], columns=cols).to_csv(output_path, index=False)

    ################################# topic propagation #########################################
    # topic_treshold = 0.0
    # num_topics = 2
    # tweet_path = Path('data/full_tweets/')
    # output_path = Path(f'topic_propagation_scores_{dataset_type}_dataset_LDA{num_topics}_minprob_{topic_treshold}{"" if filter_retweets else "_with_retweets"}_run1.csv')
    #
    # base_score_names = ['directed_NMI', 'directed_AMI', 'directed_ARI', 'undirected_NMI', 'undirected_AMI',
    #                     'undirected_ARI']
    # cols = ['graph_name'] + ['louvain' + s for s in base_score_names] + ['metis' + s for s in base_score_names] + ['rsc' + s for s in base_score_names]
    # # cols = ['graph_name'] + ['louvain' + s for s in base_score_names]
    #
    # if output_path.exists():
    #     processed_graphs = pd.read_csv(output_path)['graph_name'].tolist()
    # else:
    #     processed_graphs = []
    #
    # graphs_n = len(list(network_path.iterdir())) - len(processed_graphs)
    # for edgelist_file in tqdm(network_path.iterdir(), desc='process graphs', total=graphs_n):
    #     if edgelist_file.name.endswith(suffix):
    #         graph_name = edgelist_file.name.replace(suffix, '').replace(prefix, '')
    #         if graph_name not in processed_graphs and (tweet_path / f'{graph_name}{tweets_file_name_suffix}').exists():
    #             print(graph_name)
    #             # G = nx.read_weighted_edgelist(edgelist_file, delimiter=',', create_using=nx.DiGraph)
    #             G = graph_read_fn(edgelist_file)
    #
    #             G.graph['edge_weight_attr'] = 'weight'
    #             edge_weight = nx.get_edge_attributes(G, 'weight')
    #             nx.set_edge_attributes(G, {k: int(v) for k, v in edge_weight.items()}, 'weight')
    #
    #             # tweets = read_tweets_from_file(tweet_path / f'{graph_name}.txt', filter_retweets=False)
    #             tweets = read_tweet_fn(tweet_path / f'{graph_name}{tweets_file_name_suffix}')
    #
    #             # louvain_scores , metis_scores , rsc_scores = topic_propagation(graph_name, G, tweets, network_type='retweet')
    #             louvain_scores = topic_propagation(graph_name, G, tweets, network_type='retweet',
    #                                                topic_treshold=topic_treshold, num_topics=num_topics,
    #                                                export_results=export_results)
    #             row = [graph_name] + louvain_scores
    #             if output_path.exists():
    #                 pd.DataFrame([row], columns=cols).to_csv(output_path, index=False, header=False, mode='a')
    #             else:
    #                 pd.DataFrame([row], columns=cols).to_csv(output_path, index=False)

    ##################################3 VMQC method #########################################
    # tweet_path = Path('data/full_tweets/')
    # output_path = Path(
    #     f'VMQC_method_scores_{dataset_type}_dataset{"" if filter_retweets else "_with_retweets"}_run2.csv')
    #
    # run_contrevercy_method_experiment(network_path, tweet_path, output_path, 'VMQC_score', vmqc_pol)
    #
    # output_path = Path(
    #     f'semantic_distance_scores_{dataset_type}_dataset{"" if filter_retweets else "_with_retweets"}_run2.csv')
    #
    # run_contrevercy_method_experiment(network_path, tweet_path, output_path, 'semantic_distance', semantic_distance_pol)

    # topic_min_prob = 0.7
    # output_path = Path(
    #     f'directed_leiden_topic_propagation_scores_{dataset_type}_dataset{"" if filter_retweets else "_with_retweets"}_minprob{topic_min_prob}_run2.csv')
    # directed_topic_propagation_pol = partial(topic_propagation_pol, directed=True, topic_min_prob=topic_min_prob)
    # run_contrevercy_method_experiment(network_path, tweet_path, output_path, ['directed_NMI', 'directed_AMI', 'directed_ARI'], directed_topic_propagation_pol)
    #
    # output_path = Path(
    #     f'undirected_leiden_topic_propagation_scores_{dataset_type}_dataset{"" if filter_retweets else "_with_retweets"}_minprob{topic_min_prob}_run2.csv')
    #
    # undirected_topic_propagation_pol = partial(topic_propagation_pol, directed=False, topic_min_prob=topic_min_prob)
    # run_contrevercy_method_experiment(network_path, tweet_path, output_path,
    #                                   ['undirected_NMI', 'undirected_AMI', 'undirected_ARI'], undirected_topic_propagation_pol)
