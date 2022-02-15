from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

scores_df = pd.read_csv('topic_propagation_scores_kiran_dataset_run3.csv')
labels_df = pd.read_csv('labels.csv')
merge_df = pd.merge(scores_df, labels_df, how="inner", left_on='graph_name',right_on='topic',)

# scores = [col for col in scores_df.columns if col.endswith('_metis') or col.endswith('_rsc')]
scores = scores_df.columns[1:]
le = LabelEncoder()
y_true = le.fit_transform(merge_df['label'])
if le.classes_[0] == 'contreversial':
    y_true = 1 - y_true
for score in scores:
    y_score = merge_df[score]
    if score.startswith('extei_') or score.startswith('ei_'):
        fpr, tpr, thresholds = metrics.roc_curve(1 - y_true, y_score)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    roc_auc_score = metrics.auc(fpr, tpr)
    print('Score:', score, ', AUC:', roc_auc_score)
print('graphs:', len(labels_df))