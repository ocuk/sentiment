import pandas as pd
import numpy as np
np.random.seed(0)
import random
random.seed(0)

import operator
import itertools
from itertools import combinations

import os
from tqdm import tqdm
import pickle

import nltk
from nltk import sent_tokenize, word_tokenize

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.model_selection import KFold

from scipy.stats import mode

from sentiment_dictionaries import SentiWordNet, SentiWS

from matplotlib import pyplot as plt


def get_max_length(x):

	max_len = 0
	for i in range(len(x)):
		if len(x.iloc[i]) > max_len:
			max_len = len(x.iloc[i])

	return max_len


def padding(score_list):

	max_num = 82
	num_to_pad = max_num - len(score_list)

	return score_list + [0]*num_to_pad


def get_feature(data, partition, name):
	
	try:
		X_train = data[partition == 'train'][name].apply(eval)
		X_devel = data[partition == 'devel'][name].apply(eval)
		X_test = data[partition == 'test'][name].apply(eval)
	except:
		X_train = data[partition == 'train'][name]
		X_devel = data[partition == 'devel'][name]
		X_test = data[partition == 'test'][name]
	
	X_train = X_train.apply(padding)
	X_devel = X_devel.apply(padding)
	X_test = X_test.apply(padding)

	X_train = np.array(X_train.to_list())
	X_devel = np.array(X_devel.to_list())
	X_test = np.array(X_test.to_list())

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_devel = scaler.transform(X_devel)
	X_test = scaler.transform(X_test)
	
	return X_train, X_devel, X_test


def get_SentiWordNet_scores(data, selected_pos_stats=False, selected_neg_stats=False):
	'''
	Returns the statistics of sentiment scores for every document in a list
	data : a pandas dataframe with columns: ['ID', 'partition', 'filename', 'English', 'German', 'Dutch']
	'''
	print('-----------------------------------')
	print(data)
	SentiWordNet_path = os.path.join(ROOT, 'dictionaries', 'SentiWordNet')
	senti_dict = SentiWordNet(SentiWordNet_path)

	tonal_scores_pos = []
	tonal_scores_neg = []
	ids = []

	# each i is one text (sample)
	for i in tqdm(range(len(data))):

		text = data['English'].iloc[i]

		# Scores for each document (num_features,)
		pos_dict = senti_dict.get_stats(text, 'PosScore', selected=selected_pos_stats)
		neg_dict = senti_dict.get_stats(text, 'NegScore', selected=selected_neg_stats)
		
		# Scores for all documents (num_documents, num_features)
		tonal_scores_pos.append(pos_dict)
		tonal_scores_neg.append(neg_dict)

	# Create a dataframe with all stats for all docs

	tonal_scores_pos = pd.DataFrame(tonal_scores_pos)
	tonal_scores_pos.columns = ['SentiWordNet_' + s + '_pos' for s in tonal_scores_pos.columns]
	tonal_scores_pos.insert(0, 'filename', list(data['filename']))
	tonal_scores_neg = pd.DataFrame(tonal_scores_neg)
	tonal_scores_neg.columns = ['SentiWordNet_' + s + '_neg' for s in tonal_scores_neg.columns]
	tonal_scores_neg.insert(0, 'filename', list(data['filename']))

	return tonal_scores_pos, tonal_scores_neg


def get_SentiWS_scores(data, selected_stats=False, pos=False):
	'''
	Returns the statistics of sentiment scores for every sentence in a list
	'''
	
	SentiWS_path = os.path.join(ROOT, 'dictionaries', 'SentiWS')
	senti_dict = SentiWS(SentiWS_path, pos)

	tonal_scores = []

	for i in tqdm(range(len(data))):

		text = data['German'].iloc[i]
		stats = []
			
		# Scores for all documents (num_documents, num_sentences, num_features)
		tonal_scores.append(senti_dict.get_stats(text, pos=False, selected=selected_stats))
	
	tonal_scores = pd.DataFrame(tonal_scores)
	tonal_scores.columns = ['SentiWS_' + s for s in tonal_scores.columns]
	tonal_scores.insert(0, 'filename', list(data['filename']))
	return tonal_scores


def find_max_length(x):

	max_val = 0
	for sequence in x:
		for i in range(len(sequence)):
			current_length = len(sequence[i])
			if current_length > max_val:
				max_val = current_length
	
	return max_val


def pad_sequences(x, pad_type='pre'):
	'''
	Arguments:
		x - list of lists [num_documents, num_sentences, num_features]
	'''
	assert pad_type in ['post', 'pre'], 'Wrong pad type!'

	max_length = find_max_length(x)
	print(max_length)

	padded_sequences = []	
	for sequence in x:
		num_feats = len(sequence[0][0])
		sequence = apply_padding(sequence, max_length=max_length, num_features=num_feats, pad_type=pad_type) 
		padded_sequences.append(sequence)
	
	return padded_sequences


def apply_padding(x, max_length, num_features, pad_type):
	'''
	Arguments:
		x - (num_documents, num_sentences, num_features)
	'''

	x_ = []

	# For every sentence:
	for i in range(len(x)):

		# Determine the size of padding for current sentence  
		num_to_pad = max_length - len(x[i])
		
		# Create a list of dummy feature sequences 
		# Each sequence element is a list of size (num_features)
		list_to_pad = []
		for j in range(num_to_pad):
			list_to_pad.append([0]*num_features)
		if pad_type == 'post':
			x_.append(x[i] + list_to_pad)
		elif pad_type == 'pre':
			x_.append(list_to_pad + x[i])

	return np.array(x_)


def read_features(data, save_path):
	'''
	Reads features from text files and looks up the dictionaries 
	Returns: lists of lists [num_samples, num_sentences, num_features]
		X_train_pos : SentiWordNet pos scores
		X_train_neg : SentiWordNet neg scores
		X_devel_pos : SentiWordNet pos scores
		X_devel_neg : SentiWordNet neg scores
		X_train : SentiWS scores
		X_devel : SentiWS scores

	'''

	# Get the dictionary scores
	X_train_pos, X_train_neg = get_SentiWordNet_scores(data[data['partition'] == 'train'])
	X_devel_pos, X_devel_neg = get_SentiWordNet_scores(data[data['partition'] == 'devel'])


	# Get SentiWS dictionary scores
	X_train = get_SentiWS_scores(data[data['partition'] == 'train'], pos=False)
	X_devel = get_SentiWS_scores(data[data['partition'] == 'devel'], pos=False)
	
	X = (X_train_pos, X_train_neg, X_devel_pos, X_devel_neg, X_train, X_devel)

	return X


def load_features(data, save_path, calculate_new):
	
	if calculate_new or not os.path.isfile(save_path):
		X = read_features(data, save_path)
	else:
		with open(save_path, 'br') as f:
			X = pickle.load(f)	

	X_train_pos, X_train_neg, X_devel_pos, X_devel_neg, X_SentiWS_train, X_SentiWS_devel = X

	for i, j in enumerate(X):
		print(i, j)

	X_SentiWordNet_train = pd.merge(X_train_pos, X_train_neg, on='filename')
	X_SentiWordNet_devel = pd.merge(X_devel_pos, X_devel_neg, on='filename')
	
	X_SentiWordNet_train.to_csv('SentiWordNet_features_train.csv')
	X_SentiWordNet_devel.to_csv('SentiWordNet_features_devel.csv')

	X_SentiWS_train.to_csv('SentiWS_features_train.csv')
	X_SentiWS_devel.to_csv('SentiWS_features_devel.csv')
	
	X_train = pd.merge(X_SentiWordNet_train, X_SentiWS_train, on='filename')
	X_devel = pd.merge(X_SentiWordNet_devel, X_SentiWS_devel, on='filename')

	return X_train, X_devel


def evaluate(X_train, X_devel, y_train, y_devel):

	# Define classifiers to use
	lr_solvers = ['newton-cg', 'lbfgs', 'liblinear']
	svm_solvers = ['linear', 'poly', 'rbf', 'sigmoid']
	solvers = lr_solvers

	scores_train = pd.DataFrame(columns=solvers)
	scores_devel = pd.DataFrame(columns=solvers)

	scores_train['C'] = np.concatenate([np.array([0.00001, 0.0001, 0.001]), np.around(np.arange(0.01, 1.0, 0.01), decimals=2), np.around(np.arange(1.0, 10.01, 1), decimals=2)])
	scores_devel['C'] = np.concatenate([np.array([0.00001, 0.0001, 0.001]), np.around(np.arange(0.01, 1.0, 0.01), decimals=2), np.around(np.arange(1.0, 10.01, 1), decimals=2)])

	scores_train.set_index('C', inplace=True)
	scores_devel.set_index('C', inplace=True)

	best_devel_score = 0
	best_devel_classifier = None
	best_overall_score = 0
	best_overall_classifier = None
	best_harmonic_mean_score = 0
	best_harmonic_mean_classifier = None
	
	for solver in tqdm(solvers):
		for i in list(scores_train.index):
		
			if solver in svm_solvers:
				clf = SVC(C=i, kernel=solver, random_state=0).fit(X_train, y_train)
			elif solver in lr_solvers:
				clf = LogisticRegression(C=i, solver=solver, random_state=0, max_iter=1000).fit(X_train, y_train)
		
			train_uar = np.around(recall_score(y_train, clf.predict(X_train), average='macro'), decimals=4)
			devel_uar = np.around(recall_score(y_devel, clf.predict(X_devel), average='macro'), decimals=4)
			overall_score = np.mean([train_uar, devel_uar]) - abs(train_uar - devel_uar)
			harmonic_mean_score = (2*train_uar*devel_uar) / (train_uar + devel_uar)

			# Define best devel performance
			if devel_uar > best_devel_score:
				best_devel_score = devel_uar
				best_devel_classifier = clf

			# Define best overall performance				
			if overall_score > best_overall_score:
				best_overall_score = overall_score
				best_overall_classifier = clf

			# Define best harmonic mean classifier
			if harmonic_mean_score > best_harmonic_mean_score:
				best_harmonic_mean_score = harmonic_mean_score
				best_harmonic_mean_classifier = clf

			scores_train.loc[i, solver] = train_uar
			scores_devel.loc[i, solver] = devel_uar
	
	# Find the best result
	# best_devel_score = scores_devel.max().max()
	optimal_clf = pd.to_numeric(scores_devel.max()).idxmax()
	optimal_C = scores_devel[optimal_clf].astype(float).idxmax()
	# with open('log.txt', 'a') as handle:
	# 	handle.write(columns + ', ' + str(best_devel_score) + '\n')

	scores = (scores_train, scores_devel, best_devel_score, best_overall_score, best_harmonic_mean_score)
	classifiers = (best_devel_classifier, best_overall_classifier, best_harmonic_mean_classifier)
	return scores, classifiers


def find_best_features(X_train, X_devel, y_train, y_devel):

	feature_scores_test = {}
	feature_scores_overall = {}
	num_features = 2
	for j in tqdm(combinations(range(19), num_features)):
		X_train_part = X_train[:, j]
		X_devel_part = X_devel[:, j]

		columns = ', '.join([str(a) for a in j])
		scores, classifiers = evaluate(X_train_part, X_devel_part, y_train, y_devel, columns)
		
		scores_train, scores_devel, best_devel_score, best_overall_score = scores
		best_devel_classifier, best_overall_classifier = classifiers

		feature_scores_test[columns] = best_devel_score
		feature_scores_overall[columns] = best_overall_score

	return feature_scores_test, feature_scores_overall


def plot():

	f1, f2 = 1, 0
	figure_count = 0
	for j in range(0, 19, 5):
		# plt.figure(figure_count)
		fig, axs = plt.subplots(5, 2)
		for i in range(j, j+5):
			if i == 19:
				break
			axs[i-j][0].scatter(X_train[:, f1], X_train[:, i], c=y_train)
			scatter = axs[i-j][1].scatter(X_devel[:, f1], X_devel[:, i], c=y_devel)
		
		legend1 = fig.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
		fig.add_artist(legend1)
		fig.suptitle("Feature " + str(f1) + ' against ' + str(f2))
		figure_count += 1
		plt.show()

		# fig, axs = plt.subplots(4, 5)
		
		# f1, f2 = 5, 1
		# for i in range(4):
		# 	for j in range(4):
		# 		axs[i][j].scatter(X_train[:, f1], X_train[:, f2], c=y_train)
		# 		axs[i][j].scatter(X_devel[:, f1], X_devel[:, f2], c=y_devel)
		# 		axs[i][j].set_xlabel(f2)
		# 		f2 += 1

		# for k in range(2):
		# 	axs[k][4].scatter(X_train[:, f1], X_train[:, f2], c=y_train)
		# 	axs[k][4].scatter(X_devel[:, f1], X_devel[:, f2], c=y_devel)
		# 	f2 += 1

		# fig.suptitle("Feature " + str(f1))
		# plt.show()


def load_data():
	data = pd.read_csv(os.path.join(ROOT, 'transcriptions.csv'))
	def func(x):
		id_s = str(x['ID'])
		if len(id_s) < 4:	# there are two ids with leading zeros that get lost 
			id_s = '0' + id_s
		return x['partition'] + '_' + id_s + '.wav'
	data['filename'] = data.apply(lambda row: func(row), axis=1)

	y_train = pd.read_csv(os.path.join(ROOT, 'train.csv'))
	y_devel = pd.read_csv(os.path.join(ROOT, 'devel.csv'))
	labels = pd.concat([y_train, y_devel], axis=0)
	data = data.merge(labels, on='filename')
	return data

if __name__ == '__main__':

	# Define paths
	ROOT = r'C:\Users\User\Projects\ComParE-2021'
	data_path = r'C:\Users\User\Projects\ComParE-2021\transcriptions.csv'
	save_path = r'C:\Users\User\Projects\ComParE-2021\X_full.pickle'
	
	# Use once when running the first time
	# nltk.download('punkt')
	# nltk.download('averaged_perceptron_tagger')

	# Loading data
	data = load_data()
	
	# Read or load dictionary features
	# X_train, X_devel = load_features(data, save_path, calculate_new=True)

	# Read labels
	X_train = pd.read_csv('X_train_dict_feats.csv')
	X_devel = pd.read_csv('X_devel_dict_feats.csv')
	y_train = data['label'][data['partition'] == 'train']
	y_devel = data['label'][data['partition'] == 'devel']

	with open('best_harmonic_mean_log.csv', 'w') as handle:
		handle.write('features, train, devel, harmonic\n')

	with open('best_devel_log.csv', 'w') as handle:
		handle.write('features, train, devel\n')

	# Look through combinations of features
	for i in [6]:
		for j in combinations(range(1, 19), i):
			print(j)

			X_train_subset = X_train.drop('filename', axis=1).iloc[:, list(j)]
			X_devel_subset = X_devel.drop('filename', axis=1).iloc[:, list(j)]

			# Normalize the data
			scaler = StandardScaler()
			scaler = scaler.fit(X_train_subset)

			X_train_subset = scaler.transform(X_train_subset)
			X_devel_subset = scaler.transform(X_devel_subset)

			# Run a single train/devel experiment
			scores, classifiers = evaluate(X_train_subset, X_devel_subset, y_train, y_devel)
			scores_train, scores_devel, best_devel_score, best_overall_score, best_harmonic_mean_score = scores
			best_devel_classifier, best_overall_classifier, best_harmonic_mean_classifier = classifiers

			# Log the results
			with open('best_devel_log.csv', 'a') as handle:
				train_uar = np.around(recall_score(y_train, best_devel_classifier.predict(X_train_subset), average='macro'), decimals=4)
				devel_uar = np.around(recall_score(y_devel, best_devel_classifier.predict(X_devel_subset), average='macro'), decimals=4)
				handle.write('"' + ','.join(list(map(lambda x: str(x), j))) + '"' + ',' +  str(train_uar) + ',' + str(devel_uar) + '\n')

			with open('best_harmonic_mean_log.csv', 'a') as handle:
				train_uar = np.around(recall_score(y_train, best_harmonic_mean_classifier.predict(X_train_subset), average='macro'), decimals=4)
				devel_uar = np.around(recall_score(y_devel, best_harmonic_mean_classifier.predict(X_devel_subset), average='macro'), decimals=4)
				harmonic_mean =  np.around((2*train_uar*devel_uar) / (train_uar + devel_uar), decimals=4)
				handle.write('"' + ','.join(list(map(lambda x: str(x), j))) + '"' + ',' +  str(train_uar) + ',' + str(devel_uar) + ',' + str(harmonic_mean) + '\n')
