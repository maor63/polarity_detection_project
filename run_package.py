import os

os.environ["METIS_DLL"] = "polarity_quntification/dll/metis.dll"

import networkx as nx
import polarity_quntification.polarization_algorithms.partition_algorithms as pa


G = nx.read_gml('data/gml_files/baltimore.gml')
G = nx.to_undirected(G)
G.graph['edge_weight_attr'] = 'weight'
edge_weight = nx.get_edge_attributes(G, 'weight')
nx.set_edge_attributes(G, {k: int(v) for k, v in edge_weight.items()}, 'weight')


ms_rsc = pa.partition_spectral(G, 3)

T_rsc = [node for node in ms_rsc if ms_rsc[node] == 0]
S_rsc = [node for node in ms_rsc if ms_rsc[node] == 1]
print("RSC completed.")

ms_metis = pa.partition_metis(G)
T_metis = [node for node in ms_metis if ms_metis[node] == 0]
S_metis = [node for node in ms_metis if ms_metis[node] == 1]
print("METIS completed.")

n_sim, n_walks = 5, int(1e4)

rwc_rsc = pol.random_walk_pol(G, ms_rsc, 10, n_sim, n_walks)
rwc_metis = pol.random_walk_pol(G, ms_metis, 10, n_sim, n_walks)
print("RWC nonadaptive completed.")