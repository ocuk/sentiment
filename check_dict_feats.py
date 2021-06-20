import pandas as pd
import numpy as np
import os 
from run_sentiment import evaluate

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import recall_score, confusion_matrix
import re
from collections import Counter
from itertools import combinations

import pickle


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
	lavels_train = data[['label', 'filename']][data['partition'] == 'train']
	# lavels_train['filename'] = lavels_train['filename'].apply(lambda x: x.split('.')[0])
	lavels_devel = data[['label', 'filename']][data['partition'] == 'devel']
	# lavels_devel['filename'] = lavels_devel['filename'].apply(lambda x: x.split('.')[0])
	labels = pd.concat([lavels_train, lavels_devel])

	X_train = pd.read_csv(rf'X_train_dict_feats.csv')
	X_devel = pd.read_csv(rf'X_devel_dict_feats.csv')
	X = pd.concat([X_train, X_devel], axis=0)

	result_best_devel = {}
	result_best_harm = {}
	result_best_devel_train = {}
	result_best_harm_train = {}
	for comb in combinations([2,3,4,5,6,7,10,11,16,17,18], 5):

		mean_cv_uar_best_devel = []
		mean_cv_uar_best_harm = []
		mean_cv_uar_best_devel_train = []
		mean_cv_uar_best_harm_train = []
		for fold in range(1, 5):
			print('Fold ', fold)

			devel_files = pd.read_csv(os.path.join(ROOT, 'folds', f'data_fold_{fold}.csv'))['filename'].values
			X_devel = X[X['filename'].apply(lambda x: x.split('.')[0]).isin(devel_files)]

			
			X_train = X[~X['filename'].apply(lambda x: x.split('.')[0]).isin(devel_files)]

			X_train = X_train.merge(labels, on='filename')
			X_devel = X_devel.merge(labels, on='filename')

			
			y_devel = X_devel['label']
			y_train = X_train['label']

			# fs = [2,4,7,10,11,16]
			# fs = [3,6,11,16,17,18]
			fs = [3,5,6,16,17]
			best_devel_uars = []
			best_devel_uars_train = []
			best_harmonic_mean_uars = []
			best_harmonic_mean_uars_train = []
			dict_feats = list(map(lambda x: x+1, fs))
			dict_feats.insert(0, 0) # to include the filename at the beginning
		
			X_train_subset = X_train.drop(['label'], axis=1).iloc[:, dict_feats]
			X_devel_subset = X_devel.drop(['label'], axis=1).iloc[:, dict_feats]

			X_train_subset = X_train_subset.drop('filename', axis=1)
			X_devel_subset = X_devel_subset.drop('filename', axis=1)


			# # # Normalize the data
			scaler = StandardScaler()
			scaler = scaler.fit(X_train_subset)

			X_train_subset = scaler.transform(X_train_subset)
			X_devel_subset = scaler.transform(X_devel_subset)


			scores, classifiers = evaluate(X_train_subset, X_devel_subset, y_train, y_devel)
			scores_train, scores_devel, best_devel_score, best_overall_score, best_harmonic_mean_score = scores
			best_devel_classifier, best_overall_classifier, best_harmonic_mean_classifier = classifiers


			train_uar = np.around(recall_score(y_train, best_devel_classifier.predict(X_train_subset), average='macro'), decimals=4)
			devel_uar = np.around(recall_score(y_devel, best_devel_classifier.predict(X_devel_subset), average='macro'), decimals=4)

			name = ','.join(map(lambda x: str(x), list(comb)))

			with open('cv_log_best_devel.txt', 'a') as handle:
				handle.write('"' + name + '"' + ',' +  str(train_uar) + ',' + str(devel_uar) + '\n')

			# best_devel_uars.append(devel_uar)
			# best_devel_uars_train.append(train_uar)

			# preds = pd.DataFrame({'0': best_devel_classifier.predict(X_devel_subset), 'filename': X_devel['filename']})
			# preds['filename'] = preds['filename'].apply(lambda x: x.split('.')[0])
			# preds.to_csv(rf'C:\Users\User\Projects\ComParE-2021\predictions\preds_devel_dict_5_best.csv', index=None)
			# print(train_uar, devel_uar)
			mean_cv_uar_best_devel.append(devel_uar)
			mean_cv_uar_best_devel_train.append(train_uar)


			train_uar = np.around(recall_score(y_train, best_harmonic_mean_classifier.predict(X_train_subset), average='macro'), decimals=4)
			devel_uar = np.around(recall_score(y_devel, best_harmonic_mean_classifier.predict(X_devel_subset), average='macro'), decimals=4)

			with open('cv_log_best_harm.txt', 'a') as handle:
				handle.write('"' + name + '"' + ',' +  str(train_uar) + ',' + str(devel_uar) + '\n')

			best_harmonic_mean_uars.append(devel_uar)
			best_harmonic_mean_uars_train.append(train_uar)
			# print(train_uar, devel_uar)
			mean_cv_uar_best_harm.append(devel_uar)
			mean_cv_uar_best_harm_train.append(train_uar)

		
		result_best_devel[name] = np.mean(mean_cv_uar_best_devel)
		result_best_harm[name] = np.mean(mean_cv_uar_best_harm)
		result_best_devel_train[name] = np.mean(mean_cv_uar_best_devel_train)
		result_best_harm_train[name] = np.mean(mean_cv_uar_best_harm_train)

		print(name, np.around(np.mean(mean_cv_uar_best_devel), decimals=4), np.around(np.mean(mean_cv_uar_best_harm), decimals=4))



	print()
	print(f'Best devel: devel - {np.around(max(result_best_devel.values()), decimals=4)}, train - {np.around(result_best_devel_train[max(result_best_devel, key=result_best_devel.get)], decimals=4)}, {max(result_best_devel, key=result_best_devel.get)}')
	print(f'Best harmo: devel - {np.around(max(result_best_harm.values()), decimals=4)}, train - {np.around(result_best_harm_train[max(result_best_harm, key=result_best_harm.get)], decimals=4)}, {max(result_best_harm, key=result_best_harm.get)}')
			# with open(f'best_dict_classifier_247101116_fold_{fold}', 'bw') as handle:
			# 	pickle.dump(best_devel_classifier, handle)

			# df = pd.DataFrame({'prediction': best_devel_classifier.predict(X_devel_subset)})
			# cols = list(df.columns)
			# cols.insert(0, 'filename')
			# df['filename'] = X_devel['filename']
			# df = df[cols]
			# print(df)
			# df.to_csv(f'best_dict_classifier_247101116_fold_{fold}_val_preds.csv', index=None)
			

		# print()
		# print(f'Best devel: {np.mean(best_devel_uars), np.mean(best_devel_uars_train)}\nBest harmonic mean: {np.mean(best_harmonic_mean_uars), np.mean(best_harmonic_mean_uars_train)}')


		# 	# # # preds = pd.DataFrame({'0': best_devel_classifier.predict(X_devel_subset), 'filename': X_devel['filename']})
		# 	# # # preds['filename'] = preds['filename'].apply(lambda x: x.split('.')[0])
		# 	# # # preds.to_csv('preds_devel_{filename}_best_harmonic', index=None)
