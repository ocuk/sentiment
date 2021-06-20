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
	feat_dir = os.path.join(ROOT, 'features', 'Baseline', 'deepspectrum')

	# data = load_data()
	# print(data)
	# lavels_train = data[['label', 'filename']][data['partition'] == 'train']
	# lavels_train['filename'] = lavels_train['filename'].apply(lambda x: x.split('.')[0])
	# lavels_devel = data[['label', 'filename']][data['partition'] == 'devel']
	# lavels_devel['filename'] = lavels_devel['filename'].apply(lambda x: x.split('.')[0])

	# for file_name in os.listdir(os.path.join(ROOT, 'features', 'Dutch')):
	
	# for book_size in [2000]:
	# 	for noise in [50]:

	X_train = pd.read_csv(os.path.join(feat_dir, 'train.csv'))
	X_devel = pd.read_csv(os.path.join(feat_dir, 'devel.csv'))
	# X_test = pd.read_csv(os.path.join(feat_dir, str(book_size), str(noise), 'test.csv'))

	# X_train = pd.read_csv('X_train_dict_feats.csv')
	# X_devel = pd.read_csv('X_devel_dict_feats.csv')

	X = pd.concat([X_train, X_devel], axis=0)

	print(X)

	# X_test = pd.read_csv(os.path.join(feat_dir, str(book_size), str(noise), 'test.csv'))
	# for comb in combinations([2,3,4,5,6,7,10,11,16,17,18], 5):

	result = {}
	result[50] = []
	result[150] = []
	result[250] = []
	for fold in range(1, 5):

		devel_files = pd.read_csv(os.path.join(ROOT, 'folds', f'data_fold_{fold}.csv'))['filename'].values
		X_devel = X[X['name'].apply(lambda x: x.split('.')[0]).isin(devel_files)]
		X_train = X[~X['name'].apply(lambda x: x.split('.')[0]).isin(devel_files)]

		y_devel = X_devel['label']
		y_train = X_train['label']

		# # If labels are not included in the features
		# X_train = X_train.merge(lavels_train, on='filename')
		# X_devel = X_devel.merge(lavels_devel, on='filename')
		# y_train = X_train['label']
		# y_devel = X_devel['label']

		# For dict features
		# dict_feats = list(comb)
		# X_train_subset = X_train.drop(['name', 'label'], axis=1).iloc[:, dict_feats]
		# X_devel_subset = X_devel.drop(['name', 'label'], axis=1).iloc[:, dict_feats]

		# Normalize the data
		scaler = StandardScaler()
		# scaler_test = StandardScaler()
		scaler = scaler.fit(X_train.drop(['name', 'label'], axis=1))

		X_train_subset_ = scaler.transform(X_train.drop(['name', 'label'], axis=1))
		X_devel_subset_ = scaler.transform(X_devel.drop(['name', 'label'], axis=1))
		
		# X_test_subset = scaler_test.fit_transform(X_test.drop(['name', 'class'], axis=1))

		for n in [250]:
			pca = PCA(n_components=n)
			pca.fit(X_train_subset_)

			X_train_subset = pca.transform(X_train_subset_)
			X_devel_subset = pca.transform(X_devel_subset_)
			# X_test_subset = pca.fit_transform(X_test_subset_)

			scores, classifiers = evaluate(X_train_subset, X_devel_subset, y_train, y_devel)
			scores_train, scores_devel, best_devel_score, best_overall_score, best_harmonic_mean_score = scores
			best_devel_classifier, best_overall_classifier, best_harmonic_mean_classifier = classifiers

			train_uar = np.around(recall_score(y_train, best_devel_classifier.predict(X_train_subset), average='macro'), decimals=4)
			devel_uar = np.around(recall_score(y_devel, best_devel_classifier.predict(X_devel_subset), average='macro'), decimals=4)
			# result[n].append(devel_uar)
			# print(f'Fold {fold} best devel classifier:\ntrain: {train_uar}\ndevel: {devel_uar}')

			# test_preds = best_devel_classifier.predict_proba(X_test_subset)
			devel_preds = best_devel_classifier.predict_proba(X_devel_subset)

			# test_preds = pd.DataFrame(test_preds, columns=best_devel_classifier.classes_)
			# test_preds['filename'] = X_test['name']
			# test_preds = test_preds[['filename', 0, 1, 2]]
			# print(test_preds)

			devel_preds = pd.DataFrame(devel_preds, columns=best_devel_classifier.classes_)
			devel_preds['filename'] = X_devel['name'].values
			devel_preds['label'] = y_devel.values
			devel_preds = devel_preds[['filename', 'label', 0, 1, 2]]
			print(devel_preds)
			
			# devel_preds = pd.DataFrame({'filename': X_devel['name'], f'DeepSpectrum_{n}': devel_preds})

			# if not os.path.isdir(rf'C:\Users\User\Projects\ComParE-2021\predictions\Test\Boaw_{book_size}_{noise}'):
			# 	os.makedirs(rf'C:\Users\User\Projects\ComParE-2021\predictions\Test\Boaw_{book_size}_{noise}')
			# test_preds.to_csv(rf'C:\Users\User\Projects\ComParE-2021\predictions\Test\Boaw_{book_size}_{noise}\df_test_prob_{fold}.csv', index=None)

			if not os.path.isdir(rf'C:\Users\User\Projects\ComParE-2021\predictions\Devel\CV\Single\DeepSpectrum_{n}'):
				os.makedirs(rf'C:\Users\User\Projects\ComParE-2021\predictions\Devel\CV\Single\DeepSpectrum_{n}')
			devel_preds.to_csv(rf'C:\Users\User\Projects\ComParE-2021\predictions\Devel\CV\Single\DeepSpectrum_{n}\df_dev_prob_{fold}.csv', index=None)

			# devel_preds.to_csv(rf'C:\Users\User\Projects\ComParE-2021\predictions\Devel\CV\Single\DeepSpectrum_{n}\df_dev_pred_{fold}.csv', index=None)

	# 		preds = pd.DataFrame({'0': best_devel_classifier.predict(X_devel_subset), 'filename': X_devel['name']})
	# 		preds['filename'] = preds['filename'].apply(lambda x: x.split('.')[0])
	# 		preds.to_csv(rf'C:\Users\User\Projects\ComParE-2021\predictions\preds_devel_opensmile_pca_{n}_fold_{fold}.csv', index=None)
	# print(train_uar, devel_uar)

	# 	# preds = pd.DataFrame({'0': best_devel_classifier.predict(X_devel_subset), 'filename': X_devel['filename']})
	# 	# preds['filename'] = preds['filename'].apply(lambda x: x.split('.')[0])
	# 	# preds.to_csv('preds_devel_{filename}_best_harmonic', index=None)



	# scores, classifiers = evaluate(X_train_subset, X_devel_subset, y_train, y_devel)
	# 	scores_train, scores_devel, best_devel_score, best_overall_score, best_harmonic_mean_score = scores
	# 	best_devel_classifier, best_overall_classifier, best_harmonic_mean_classifier = classifiers

	# 	test_preds = best_devel_classifier.predict(X_test_subset)
	# 	devel_preds = best_devel_classifier.predict(X_devel_subset)
		
	# 	test_preds = pd.DataFrame({'filename': X_test['name'], f'DiFE_sum_sent_nn': test_preds})
	# 	devel_preds = pd.DataFrame({'filename': X_devel['name'], f'DiFE_sum_sent_nn': devel_preds})
		
	# 	test_preds.to_csv(rf'C:\Users\User\Projects\ComParE-2021\predictions\Test\DiFE_sum_sent_nn_preds_fold_{fold}.csv', index=None)
	# 	devel_preds.to_csv(rf'C:\Users\User\Projects\ComParE-2021\predictions\Devel\CV\Single\DiFE_sum_sent_nn\df_dev_pred_{fold}.csv', index=None)

