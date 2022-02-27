'''
This is a Python implementation of the method presented in "Vocabulary-based Method for Quantifying
Controversy in Social Media" https://arxiv.org/pdf/2001.09899.pdf.
The implementation is based on the original R code https://github.com/jmanuoz/Vocabulary-based-Method-for-Quantify-Controversy
'''
import json
import time
from functools import partial
import fasttext
import community as community_louvain
from collections import Counter
import pandas as pd
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import networkx as nx
import os
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from tqdm.auto import tqdm


def vmqc_pol(G, tweets, get_tweet_username, verbos=False, graph_name='none'):
    G_copy = G.copy().to_undirected()
    output_path = Path('output/VMQC_method/')
    if not output_path.exists():
        os.makedirs(output_path)

    louvain_partition = community_louvain.best_partition(G_copy)
    largest_communities = Counter(louvain_partition.values()).most_common(2)
    community1 = largest_communities[0][0]
    community2 = largest_communities[1][0]
    user_text = create_user_text_df(get_tweet_username, louvain_partition, tweets)

    train_file_path = str(output_path / f'{graph_name}-train.txt')
    prepare_train_file(community1, community2, train_file_path, user_text)

    Gcc = sorted(nx.connected_components(G_copy), key=len, reverse=True)
    G_Giant = G_copy.subgraph(Gcc[0])
    isolated_nodes = [n for n, d in G_Giant.degree() if d < 3]
    G_Giant = G_Giant.copy()
    G_Giant.remove_nodes_from(isolated_nodes)


    user_text_G_giant = user_text[user_text['username'].isin(set(G_Giant.nodes()))]
    predict_text = clean_text_r_code(user_text_G_giant['text'])

    fasttext_model = fasttext.train_supervised(train_file_path, epoch=20, dim=200, wordNgrams=2, ws=5)

    predicted_df = predict_text_community(fasttext_model, graph_name, output_path, predict_text)

    # node_set = set(G_Giant.nodes())
    # all([(n in node_set) for n in user_text_G_giant['username']])

    # print(user_text_G_giant.shape, predicted_df.shape)
    user_text_G_giant.loc[:, 'pred'] = predicted_df.loc[:, 'pred']
    user_text_G_giant.loc[:, 'pred_prob'] = predicted_df.loc[:, 'pred_prob']
    # user_text_G_giant = pd.concat([user_text_G_giant, predicted_df], axis=1)
    # print(user_text_G_giant.shape, predicted_df.shape)
    print('Max prob:', user_text_G_giant['pred_prob'].max())
    low_prob_nodes = user_text_G_giant['pred_prob'] < 0.9
    lable_2_nodes = user_text_G_giant['pred'] == '__label__2'
    user_text_G_giant.loc[lable_2_nodes, 'pred_prob'] = -user_text_G_giant.loc[lable_2_nodes, 'pred_prob']
    user_text_G_giant.loc[low_prob_nodes, 'pred_prob'] = 0.0

    ideos = dict(zip(user_text_G_giant['username'], user_text_G_giant['pred_prob'].fillna(0.0)))
    for n in G_Giant.nodes():
        if n not in ideos:
            ideos[n] = 0.0
    node_ideos = [ideos.get(n, 0) for n in G_Giant.nodes()]
    node_to_id = dict(zip(G_Giant.nodes(), range(len(G_Giant.nodes()))))

    nx.set_node_attributes(G_Giant, ideos, name='ideo')
    nx.set_node_attributes(G_Giant, node_to_id, name='label')
    nx.set_node_attributes(G_Giant, node_to_id, name='label2')
    # ideos = nx.get_node_attributes(G, 'ideo')

    corenode = []
    for key in ideos.keys():
        if (ideos[key] == 1 or ideos[key] == -1):
            corenode.append(node_to_id[key])

    v_current = propagation_model(G_Giant, corenode)
    score = GetPolarizationIndex(v_current)
    print(score)
    if score == np.nan:
        score = 0
    return score


def predict_text_community(fasttext_model, graph_name, output_path, predict_text, output_csv=True):
    rows = []
    for txt in predict_text:
        pred = fasttext_model.predict(txt)
        rows.append((pred[0][0], pred[1][0]))  # (label, prob)
    predicted_df = pd.DataFrame(rows, columns=['pred', 'pred_prob'])
    if output_csv:
        predicted_df.to_csv(str(output_path / f'{graph_name}-predict.txt'), index=False, header=None)
    return predicted_df


def create_user_text_df(get_tweet_username, louvain_partition, tweets):
    user_names = [get_tweet_username(t) for t in tweets]
    texts = [t['text'] for t in tweets]
    user_text = pd.DataFrame(zip(user_names, texts), columns=['username', 'text'])
    user_text['community'] = user_text['username'].map(louvain_partition)
    user_text = user_text.groupby(['username', 'community'], as_index=False).agg({'text': ' '.join})
    return user_text


def prepare_train_file(community1, community2, train_file_path, user_text):
    community1_user_text = user_text[user_text['community'] == community1]
    community1_text = community1_user_text['text']
    community1_text = clean_text_r_code(community1_text)

    community2_user_text = user_text[user_text['community'] == community2]
    community2_text = community2_user_text['text']
    community2_text = clean_text_r_code(community2_text)

    community1_df = pd.DataFrame(community1_text, columns=['text'])
    community1_df['label'] = '__label__1'
    community1_df['username'] = community1_user_text['username'].tolist()

    community2_df = pd.DataFrame(community2_text, columns=['text'])
    community2_df['label'] = '__label__2'
    community2_df['username'] = community2_user_text['username'].tolist()

    communities_clean_text = pd.concat([community1_df, community2_df], axis=0)[['label', 'text']]
    communities_clean_text.to_csv(train_file_path, index=False, header=None, sep=' ')
    return community1_df, community2_df


def clean_text_r_code(community_text):
    # community_text = community_text.drop_duplicates()
    textclean_r_package = importr('textclean')
    community_text_strvector = robjects.StrVector(community_text.tolist())
    community_text_strvector = robjects.r['replace_emoji'](community_text_strvector)
    community_text_strvector = robjects.r['tolower'](community_text_strvector)
    community_text_strvector = robjects.r['gsub']("rt", " ",
                                                  community_text_strvector)  # Remove the "RT" (retweet) so duplicates are duplicates
    community_text_strvector = robjects.r['gsub']("@\\w+", " ",
                                                  community_text_strvector)  # Remove user names (all proper names if you're wise!)
    community_text_strvector = robjects.r['gsub']("http.+ |http.+$", " ", community_text_strvector)  # Remove links
    community_text_strvector = robjects.r['gsub']("[[:punct:]]", " ", community_text_strvector)  # Remove punctuation
    community_text_strvector = robjects.r['gsub']("[ |\t]{2,}", " ", community_text_strvector)  # Remove tabs
    community_text_strvector = robjects.r['gsub']("^ ", "", community_text_strvector)  # Leading blanks
    community_text_strvector = robjects.r['gsub'](" $", "", community_text_strvector)  # Lagging blanks
    community_text_strvector = robjects.r['gsub'](" +", " ",
                                                  community_text_strvector)  # General spaces (should just do all whitespaces no?)
    community_text_strvector = robjects.r['gsub']("\n", " ",
                                                  community_text_strvector)  # General spaces (should just do all whitespaces no?)
    community_text = pd.Series(list(community_text_strvector))
    return community_text


def read_tweet_fn(f_file):
    with open(f_file, encoding='latin-1') as f:
        data = json.load(f)
    return data


def propagation_model(G, corenode, tol=10 ** -5, save_xi=True):
    # G: Graph to calculate opinions. The nodes have an attribute "ideo" with their ideology, set to 0 for all listeners, 1 and -1 for the elite.
    # corenode: Nodes that belong to the seed (Identifiers from the Graph G)
    # tol is the threshold for convergence. It will evaluate the difference between opinions at time t and t+1
    # save_xi: boolean to save results

    N = len(G.nodes())
    #    print N

    # Get de adjacency matrix
    # Aij = sp.lil_matrix((N, N))
    # #    print Aij.shape
    # for o, d in G.edges():
    #     Aij[o, d] = 1
    Aij = nx.adjacency_matrix(G)

    # Build the vectors with users opinions
    macro_v_current = []
    v_current = []
    v_new = []
    dict_nodes = {}
    for nodo in G.nodes():
        dict_nodes[G.nodes[nodo]['label']] = G.nodes[nodo]['ideo']
        v_current.append(G.nodes[nodo]['ideo'])
        v_new.append(0.0)

    v_current = 1. * np.array(v_current)
    #    f2 = open("analyze_venezuela/results_iter_0","wb");
    #   pickle.dump(dict_nodes,f2);
    #    f2.close();
    v_new = 1. * np.array(v_new)

    notconverged = len(v_current)
    times = 0

    # Do as many times as required for convergence
    for i in tqdm(range(min(len(v_current), 200)), desc='label propagation'):
        if notconverged == 0:
            break
        times = times + 1
        # print (sys.stderr, times)
        t = time.time()

        # for all nodes apart from corenode, calculate opinion as average of neighbors
        for j in np.setdiff1d(range(len(v_current)), corenode):
            nodosin = Aij.getrow(j).nonzero()[1]
            if len(nodosin) > 0:
                v_new[j] = np.mean(v_current[nodosin])
            else:
                v_new[j] = v_current[j]
        #            nodos_changed[j]=nodos_changed[j] or (v_new[j]!=v_current[j])

        # update opinion
        for j in corenode:
            v_new[j] = v_current[j]

        diff = np.abs(v_current - v_new)
        notconverged = len(diff[diff > tol])
        v_current = v_new.copy()
    print('\n not converged', notconverged)
    return v_current


def GetPolarizationIndex(ideos):
    # Input: Vector with individuals Xi
    # Output: Polarization index, Area Difference, Normalized Pole Distance
    D = []  # POLE DISTANCE
    AP = []  # POSSITIVE AREA
    AN = []  # NEGATIVE AREA
    cgp = []  # POSSITIVE GRAVITY CENTER
    cgn = []  # NEGATIVE GRAVITY CENTER

    ideos.sort()
    hist, edg = np.histogram(ideos, np.linspace(-1, 1.1, 50))
    edg = edg[:len(edg) - 1]
    AP = sum(hist[edg > 0])
    AN = sum(hist[edg < 0])
    AP0 = 1. * AP / (AP + AN)
    AN0 = 1. * AN / (AP + AN)
    # if AP > 0:
    cgp = sum(hist[edg > 0] * edg[edg > 0]) / sum(hist[edg > 0])
    # else:
    #     cgp = 0
    # if AN > 0:
    cgn = sum(hist[edg < 0] * edg[edg < 0]) / sum(hist[edg < 0])
    # else:
    #     cgn = 0
    D = cgp - cgn
    p1 = 0.5 * D * (1. - 1. * abs(AP0 - AN0))  # polarization index
    DA = 1. * abs(AP0 - AN0) / (AP0 + AN0)  # Areas Difference
    D = 0.5 * D  # Normalized Pole Distance
    return p1


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

    graph_name = 'ukraine'
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

    vmqc_pol(G, tweets, get_tweet_username, verbos=False, graph_name=graph_name)
