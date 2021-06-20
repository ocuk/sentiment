import pandas as pd
import numpy as np
import os 
from run_sentiment import evaluate

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import recall_score, confusion_matrix
from sklearn.decomposition import PCA



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

	data = load_data()
	data[['filename', 'label']].to_csv('labels.csv', index=None)
	# # lavels_train = data[['label', 'filename']][data['partition'] == 'train']
	# # lavels_train['filename'] = lavels_train['filename'].apply(lambda x: x.split('.')[0])
	# # lavels_devel = data[['label', 'filename']][data['partition'] == 'devel']
	# # lavels_devel['filename'] = lavels_devel['filename'].apply(lambda x: x.split('.')[0])

	# # for file_name in os.listdir(os.path.join(ROOT, 'features', 'Dutch')):
	# feat_dir = os.path.join(ROOT, 'features', 'Baseline', 'deepspectrum')
	# # for book_size in [125, 500, 2000]:
	# # 	for noise in [10, 20, 50]:

	# X_train = pd.read_csv(os.path.join(feat_dir, 'train.csv'))
	# X_devel = pd.read_csv(os.path.join(feat_dir, 'devel.csv'))
	
	# y_devel = X_devel['label']
	# y_train = X_train['label']

	# # # # # X_train = X_train.merge(lavels_train, on='filename')
	# # # # # X_devel = X_devel.merge(lavels_devel, on='filename')

	# # y_train = X_train['label']
	# # y_devel = X_devel['label']

	# # X_train_subset = X_train.drop(['name', 'label'], axis=1)
	# # X_devel_subset = X_devel.drop(['name', 'label'], axis=1)

	# # Normalize the data
	# scaler = StandardScaler()
	# scaler = scaler.fit(X_train.drop(['name', 'label'], axis=1))

	# X_train_subset_ = scaler.transform(X_train.drop(['name', 'label'], axis=1))
	# X_devel_subset_ = scaler.transform(X_devel.drop(['name', 'label'], axis=1))

	# # scores, classifiers = evaluate(X_train_subset_, X_devel_subset_, y_train, y_devel)
	# # scores_train, scores_devel, best_devel_score, best_overall_score, best_harmonic_mean_score = scores
	# # best_devel_classifier, best_overall_classifier, best_harmonic_mean_classifier = classifiers

	# # train_uar = np.around(recall_score(y_train, best_devel_classifier.predict(X_train_subset_), average='macro'), decimals=4)
	# # devel_uar = np.around(recall_score(y_devel, best_devel_classifier.predict(X_devel_subset_), average='macro'), decimals=4)
	# # print(f'No PCA best devel classifier:\ntrain: {train_uar}\ndevel: {devel_uar}')
	

	# for n in [250]:
	# 	pca = PCA(n_components=n)
	# 	pca.fit(X_train_subset_)
	# 	X_train_subset = pca.transform(X_train_subset_)
	# 	X_devel_subset = pca.transform(X_devel_subset_)

	# 	scores, classifiers = evaluate(X_train_subset, X_devel_subset, y_train, y_devel)
	# 	scores_train, scores_devel, best_devel_score, best_overall_score, best_harmonic_mean_score = scores
	# 	best_devel_classifier, best_overall_classifier, best_harmonic_mean_classifier = classifiers

	# 	train_uar = np.around(recall_score(y_train, best_devel_classifier.predict(X_train_subset), average='macro'), decimals=4)
	# 	devel_uar = np.around(recall_score(y_devel, best_devel_classifier.predict(X_devel_subset), average='macro'), decimals=4)
	# 	print(f'PCA {n} best devel classifier:\ntrain: {train_uar}\ndevel: {devel_uar}')

	# 	preds = pd.DataFrame({'0': best_devel_classifier.predict(X_devel_subset), 'filename': X_devel['name']})
	# 	preds['filename'] = preds['filename'].apply(lambda x: x.split('.')[0])
	# 	preds.to_csv(rf'C:\Users\User\Projects\ComParE-2021\predictions\preds_devel_deepspectrum_pca_{n}.csv', index=None)

	# 	# train_uar = np.around(recall_score(y_train, best_harmonic_mean_classifier.predict(X_train_subset), average='macro'), decimals=4)
	# 	# devel_uar = np.around(recall_score(y_devel, best_harmonic_mean_classifier.predict(X_devel_subset), average='macro'), decimals=4)
	# 	# print(f'\nFold {fold} best harmonic mean classifier:\ntrain: {train_uar}\ndevel: {devel_uar}')


	
	
	
	# # print(train_uar, devel_uar)

	# # # 	# preds = pd.DataFrame({'0': best_devel_classifier.predict(X_devel_subset), 'filename': X_devel['filename']})
	# # # 	# preds['filename'] = preds['filename'].apply(lambda x: x.split('.')[0])
	# # # 	# preds.to_csv('preds_devel_{filename}_best_harmonic', index=None)
