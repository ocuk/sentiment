import pandas as pd
import numpy as np
import os

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import itertools

from tqdm import tqdm, trange


if __name__ == '__main__':


	ROOT = r'C:\Users\User\Projects\ComParE-2021'

	score_type = 'basic'

	# Load the data and labels
	data = pd.read_csv('data.csv')

	# Load dictionary-based scores
	SentiWS_path = os.path.join(ROOT, 'dict_scores', 'basic', 'SentiWS_stats_2.csv')
	SentiWS_scores = pd.read_csv(SentiWS_path)

	SentiWordNet_path = os.path.join(ROOT, 'dict_scores', 'basic', 'SentiWordNet_stats_3.csv')
	SentiWordNet_scores = pd.read_csv(SentiWordNet_path)

	# Define features to use
	features = []
	stats = ['min', 'max', 'range', 'mean', 'sum', 'num']
	dicts = [
		'pos_SentiWordNet',
		'neg_SentiWordNet']

	for stat in stats:
		for d in dicts:
			features.append(stat + '_' + d)

	stats = ['min', 'max', 'range', 'mean', 'sum', 'num_pos', 'num_neg']	
	dicts = ['SentiWS']

	for stat in stats:
		for d in dicts:
			features.append(stat + '_' + d)

	# Or use a predefined set of features
	best5_v_feature_set = ['max_pos_SentiWordNet', 'sum_neg_SentiWordNet', 'min_SentiWS', 'max_SentiWS', 'num_neg_SentiWS']
	best6_v_feature_set = ['min_neg_SentiWordNet', 'max_pos_SentiWordNet', 'range_neg_SentiWordNet', 'mean_pos_SentiWordNet', 'max_SentiWS', 'num_neg_SentiWS']
	
	best5_a_feature_set = ['max_pos_SentiWordNet', 'mean_neg_SentiWordNet', 'num_pos_SentiWordNet', 'max_SentiWS', 'num_neg_SentiWS']
	best6_a_feature_set = ['min_neg_SentiWordNet', 'max_pos_SentiWordNet', 'mean_neg_SentiWordNet', 'num_pos_SentiWordNet', 'max_SentiWS', 'num_neg_SentiWS']

	# # Merge the data and dictionary-based scores
	# for df in [SentiWS_scores, SentiWordNet_scores]:
	# 	data = data.merge(df, how='outer', on='ID_story')
	# data = data.loc[:, ['ID_story', 'partition', 'V_cat'] + best5_v_feature_set]
	
	task = 'A'
	data = data.loc[:, ['ID_story', 'partition', f'{task}_cat']]

	p_feats_path = os.path.join(ROOT, 'features', 'Gizem', 'EN_nltk_sentiment_features_per_story_arousal.csv')
	polarity_features = pd.read_csv(p_feats_path)

	# # Load additional features
	# X_f1 = pd.read_csv(os.path.join(ROOT, 'features', 'Gizem', 'EN_TFIDF_features.csv'))
	# X_f1['partition'] = data['partition']
	
	data = pd.merge(data, polarity_features, on='ID_story', left_index=True, right_index=True) 

	# Define solvers for classification
	lr_solvers = ['newton-cg', 'lbfgs', 'liblinear']
	svm_solvers = ['linear', 'poly', 'rbf', 'sigmoid']
	solvers = svm_solvers + lr_solvers

	# Partition the data
	y_train = data[data['partition'] == 'train'][f'{task}_cat']
	X_train = np.array(data[data['partition'] == 'train'].drop(['partition', f'{task}_cat', 'ID_story'], axis=1))

	y_devel = data[data['partition'] == 'devel'][f'{task}_cat']
	X_devel = np.array(data[data['partition'] == 'devel'].drop(['partition', f'{task}_cat', 'ID_story'], axis=1))

	X_test = np.array(data[data['partition'] == 'test'].drop(['partition', f'{task}_cat', 'ID_story'], axis=1))

	# Normalize the data
	scaler = StandardScaler()
	scaler = scaler.fit(X_train)

	X_train = scaler.transform(X_train)
	X_devel = scaler.transform(X_devel)
	X_test = scaler.transform(X_test)

	# Make classification
	scores_train = pd.DataFrame(columns=solvers)
	scores_devel = pd.DataFrame(columns=solvers)

	scores_train['C'] = np.around(np.arange(0.01, 1.01, 0.01), decimals=2)
	scores_devel['C'] = np.around(np.arange(0.01, 1.01, 0.01), decimals=2)

	scores_train.set_index('C', inplace=True)
	scores_devel.set_index('C', inplace=True)

	for solver in tqdm(solvers):
		for i in list(scores_train.index):
			
			if solver in svm_solvers:
				clf = SVC(C=i, kernel=solver, random_state=0).fit(X_train, y_train)
			elif solver in lr_solvers:
				clf = LogisticRegression(C=i, solver=solver, random_state=0, max_iter=1000).fit(X_train, y_train)
		
			scores_train.loc[i, solver] = np.around(recall_score(y_train, clf.predict(X_train), average='macro'), decimals=4)
			scores_devel.loc[i, solver] = np.around(recall_score(y_devel, clf.predict(X_devel), average='macro'), decimals=4)
	
	# Find the best result
	best_devel_score = scores_devel.max().max()
	optimal_clf = scores_devel.max().idxmax()
	optimal_C = scores_devel[optimal_clf].astype(float).idxmax()
	
	print('Train: ', scores_train[optimal_clf][optimal_C])
	print('Devel: ', best_devel_score, optimal_clf, optimal_C)
	print() 
