from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pathlib import Path
import os

# scores_df = pd.read_csv('structural_polarity_quantification_scores_juan_dataset.csv')
# scores_df = pd.read_csv('structural_polarity_quantification_scores_kiran_dataset.csv')
# scores_df = pd.read_csv('topic_propagation_scores_juan_dataset_LDA2_minprob_0.0_run1.csv')
# scores_df = pd.read_csv('semantic_distance_scores_juan_dataset_run1.csv')
# scores_df = pd.read_csv('VMQC_method_scores_juan_dataset_run2.csv')
# scores_df = pd.read_csv('undirected_leiden_topic_propagation_scores_juan_dataset_minprob0.7_run2.csv')
# scores_df = pd.read_csv('undirected_leiden_topic_propagation_scores_kiran_dataset_minprob0.55_run2.csv')
# labels_df = pd.read_csv('labels_juan.csv')
# labels_df = pd.read_csv('labels.csv')

# mademesmile_scores_df = pd.read_csv('structural_polarity_quantification_scores_reddit_mademesmile_dataset.csv')
# politicaldisscusions_scores_df = pd.read_csv('structural_polarity_quantification_scores_reddit_politicaldisscusions_dataset.csv')

# mademesmile_scores_df = pd.read_csv('topic_propagation_scores_reddit_mademesmile_dataset.csv')
# politicaldisscusions_scores_df = pd.read_csv('topic_propagation_scores_reddit_politicaldisscusions_dataset.csv')
# scores_df = pd.concat([mademesmile_scores_df, politicaldisscusions_scores_df], axis=0)
# labels_df = pd.DataFrame()
# labels_df['topic'] = scores_df['graph_name']
# labels_df['label'] = (['non-controversial'] * len(mademesmile_scores_df)) + (['controversial'] * len(politicaldisscusions_scores_df))
#
merge_df = pd.merge(scores_df, labels_df, how="inner", left_on='graph_name', right_on='topic', )

# scores = [col for col in scores_df.columns if col.endswith('_metis') or col.endswith('_rsc')]
scores = scores_df.columns[1:]
le = LabelEncoder()
y_true = le.fit_transform(merge_df['label'])
if le.classes_[0] == 'controversial':
    y_true = 1 - y_true
rows = []
for score in scores:
    if score in {'size', 'ave_deg'}:
        continue
    y_score = merge_df[score]
    if score.startswith('extei_') or score.startswith('ei_') or score.startswith('semantic_distance'):
        fpr, tpr, thresholds = metrics.roc_curve(1 - y_true, y_score)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    roc_auc_score = metrics.auc(fpr, tpr)
    print('Score:', score, ', AUC:', roc_auc_score)
    rows.append([score, roc_auc_score])
print('graphs:', len(labels_df))

output_path = 'kiran_evaluation_scores.csv'
if not os.path.exists(output_path):
    pd.DataFrame(rows, columns=['score', 'AUC']).to_csv(output_path, index=False)
else:
    score_df = pd.read_csv(output_path)
    new_df = pd.DataFrame(rows, columns=['score', 'AUC'])
    pd.concat([score_df, new_df], axis=0).drop_duplicates('score').to_csv(output_path, index=False)