import os
from pathlib import Path
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


def compute_polarization(dG):
    G = get_giant_component(dG)

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


network_path = Path('data/networks/retweet_networks/')
suffix = '_threshold_largest_CC.txt'
prefix = 'retweet_graph_'
#output_path = Path('structural_polarity_quantification_scores_kiran_dataset.csv')
#cols = ['graph_name', 'rwc_metis', 'arwc_metis', 'ebc_metis', 'gmck_metis', 'mblb_metis', 'mod_metis', 'ei_metis',
#        'extei_metis', 'cond_metis', 'rwc_rsc', 'arwc_rsc', 'ebc_rsc', 'gmck_rsc', 'mblb_rsc', 'mod_rsc', 'ei_rsc',
#        'extei_rsc', 'cond_rsc', 'size', 'ave_deg']
#
#if output_path.exists():
#    processed_graphs = pd.read_csv(output_path)['graph_name'].tolist()
#else:
#    processed_graphs = []
#
#graphs_n = len(list(network_path.iterdir())) - len(processed_graphs)
#for edgelist_file in tqdm(network_path.iterdir(), desc='process graphs', total=graphs_n):
#    if edgelist_file.name.endswith(suffix):
#        graph_name = edgelist_file.name.replace(suffix, '').replace(prefix, '')
#        if graph_name not in processed_graphs:
#            print(graph_name)
#            G = nx.read_weighted_edgelist(edgelist_file, delimiter=',')
#            G.graph['edge_weight_attr'] = 'weight'
#            edge_weight = nx.get_edge_attributes(G, 'weight')
#            nx.set_edge_attributes(G, {k: int(v) for k, v in edge_weight.items()}, 'weight')
#
#            row = [graph_name] + compute_polarization(G)
#            if output_path.exists():
#                pd.DataFrame([row], columns=cols).to_csv(output_path, index=False, header=False, mode='a')
#            else:
#                pd.DataFrame([row], columns=cols).to_csv(output_path, index=False)


tweet_path = Path('data/full_tweets/')
output_path = Path('topic_propagation_scores_kiran_dataset_run2.csv')

base_score_names = ['directed_NMI', 'directed_AMI', 'directed_ARI', 'undirected_NMI', 'undirected_AMI', 'undirected_ARI']
cols = ['graph_name'] + ['louvain' + s for s in base_score_names]

if output_path.exists():
    processed_graphs = pd.read_csv(output_path)['graph_name'].tolist()
else:
    processed_graphs = []

graphs_n = len(list(network_path.iterdir())) - len(processed_graphs)
for edgelist_file in tqdm(network_path.iterdir(), desc='process graphs', total=graphs_n):
    if edgelist_file.name.endswith(suffix):
        graph_name = edgelist_file.name.replace(suffix, '').replace(prefix, '')
        if graph_name not in processed_graphs and (tweet_path / f'{graph_name}.txt').exists():
            print(graph_name)
            G = nx.read_weighted_edgelist(edgelist_file, delimiter=',')
            G.graph['edge_weight_attr'] = 'weight'
            edge_weight = nx.get_edge_attributes(G, 'weight')
            nx.set_edge_attributes(G, {k: int(v) for k, v in edge_weight.items()}, 'weight')
            
            tweets = read_tweets_from_file(tweet_path / f'{graph_name}.txt', filter_retweets=False) 
            
            louvain_scores = topic_propagation(graph_name, G, tweets, network_type='retweet')            
            row = [graph_name] + louvain_scores
            if output_path.exists():
                pd.DataFrame([row], columns=cols).to_csv(output_path, index=False, header=False, mode='a')
            else:
                pd.DataFrame([row], columns=cols).to_csv(output_path, index=False)


