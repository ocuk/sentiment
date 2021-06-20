import pandas as pd 
from sklearn.metrics import recall_score
import os
import numpy as np
from collections import Counter
from itertools import permutations, combinations
from sklearn.metrics import recall_score
from matplotlib import pyplot as plt
from tqdm import tqdm

def majority_vote(df):

	vote = df.astype(int).apply(lambda row: Counter(row).most_common()[0][0], axis=1)
	return vote


def majority_vote_with_rule(df):

	def apply_rule(row):

		val1 = row[0]
		val2 = row[1]

		if val1 == val2:
			return val1
		else:
			return 1

	vote = df.astype(int).apply(lambda row: apply_rule(row), axis=1)
	return vote

if __name__ == '__main__':

	pd.options.mode.chained_assignment = None  # default='warn'

	ROOT = r'C:\Users\User\Projects\ComParE-2021'

	labels = pd.read_csv('labels.csv')
	# labels['filename'] = labels.filename.apply(lambda x: x[:-4])

	pred_types = os.listdir(r'C:\Users\User\Projects\ComParE-2021\predictions\Devel\CV\Single')
	devel_files = {}
	for i in range(1, 5):
		devel_files[i] = pd.read_csv(os.path.join(ROOT, 'folds', f'data_fold_{i}.csv'))['filename'].apply(lambda x: x + '.wav').values # no .wav 

	lingo_models = ['Dict_247101116', 'German_BERT', 'Dutch_FT', 'German_W2V', 'DiFE_sum_nn', 'DiFE_sum_sent', 'DiFE_sum_sent_nn']
	audio_models = ['Lena', 'DenisD', 'DenisK', 'Alena', 'DeepSpectrum', 'DeepSpectrum_250', 'OpenSmile_50', 'OpenSmile_250', 'Boaw_125_50', 'Boaw_2000_50']
	models = lingo_models + audio_models 

	preds = {}
	for i in range(1, 5):
		preds[i] = {}
		for pred_name in pred_types:

			if pred_name not in models:
				continue

			try:
				df = pd.read_csv(os.path.join(ROOT, rf'predictions\Devel\CV\Single\\{pred_name}\df_dev_prob_{i}.csv'))
				if 'label' in list(df.columns):
					df[pred_name] = df.drop(['filename', 'label'], axis=1).apply(lambda row: np.argmax(row), axis=1)
				else:
					df[pred_name] = df.drop(['filename'], axis=1).apply(lambda row: np.argmax(row), axis=1)
				df = df[['filename', pred_name]]

			except:
				try:
					df = pd.read_csv(os.path.join(ROOT, rf'predictions\Devel\CV\Single\\{pred_name}\df_dev_pred_{i}.csv'))
					tmp_col_name = list(df.columns)
					tmp_col_name.remove('filename')
					df = df[['filename', tmp_col_name[0]]]
					df.columns = ['filename', pred_name]
				except:
					print('Couldn\'t read', pred_name)

			df['filename'] = df['filename'].apply(lambda x: x if x.endswith('.wav') else x + '.wav')
			preds[i][pred_name] = df


	# Run the CV check for each single system
	df_result = pd.DataFrame(columns=['Fold', 'UAR','Estimators'])
	max_result = pd.DataFrame(index=[1, 2, 3, 4], columns=['UAR', 'Ensemble'])
	tmp_max = 0
	for pred_name in models:

		vote_uars = [] # to store each fold's results
		for i in range(1, 5):

			preds_set = preds[i][pred_name]
			preds_set = preds_set.merge(labels, on='filename')
			uar = np.around(recall_score(preds_set['label'], preds_set[pred_name], average='macro'), decimals=4)
			vote_uars.append(uar)
		
		name = pred_name
		print(name, np.mean(vote_uars))
	# 	if np.mean(vote_uars) > tmp_max:

	# 		tmp_max = np.mean(vote_uars)
	# 		for i in range(1, 5):
	# 			max_result.loc[i, 'UAR'] = vote_uars[i-1]
	# 			max_result.loc[i, 'Ensemble'] = name
	
	# max_result['Estimators'] = 1
	# max_result = max_result.reset_index().rename({'index': 'Fold'}, axis=1)
	# max_result.to_csv(rf'results\homo_mixed_models_1.csv', index=None)

	# df_result = pd.concat([df_result, max_result])
	# df_result.to_csv(rf'results\homo_mixed_models.csv', index=None)
	# print(df_result)
	
	# # Run the CV check for each combination of systems
	# for num_estimators in range(1, len(models)):
	
	# 	print('\nNum estimators: ', num_estimators)
	# 	max_result = pd.DataFrame(index=[1, 2, 3, 4], columns=['UAR', 'Ensemble'])

	# 	result = {}
	# 	tmp_max = 0
	# 	for pred_name in models:

	# 		models_list = models.copy()
	# 		models_list.remove(pred_name)

	# 		for perm in tqdm(combinations(models_list, num_estimators)):
	# 			models_subset = list(perm)
	# 			models_subset.insert(0, pred_name)

	# 			vote_uars = [] # to store each fold's results
	# 			for i in range(1, 5):

	# 				preds_set = pd.DataFrame({'filename': devel_files[i]})
	# 				for pred_name_ in models_subset:
	# 					preds_set = preds_set.merge(preds[i][pred_name_], on='filename')
					
	# 				if num_estimators == 1:
	# 					preds_set['vote'] = majority_vote_with_rule(preds_set.drop(['filename'], axis=1))
	# 				else:
	# 					preds_set['vote'] = majority_vote(preds_set.drop(['filename'], axis=1))
					
	# 				preds_set = preds_set.merge(labels, on='filename')
	# 				uar = np.around(recall_score(preds_set['label'], preds_set['vote'], average='macro'), decimals=4)
	# 				vote_uars.append(uar)
				
	# 			name = ', '.join(models_subset)
	# 			result[name] = np.mean(vote_uars)

	# 			if np.mean(vote_uars) > tmp_max:

	# 				tmp_max = np.mean(vote_uars)
	# 				for i in range(1, 5):
	# 					max_result.loc[i, 'UAR'] = vote_uars[i-1]
	# 					max_result.loc[i, 'Ensemble'] = name
		
	# 	max_result['Estimators'] = num_estimators + 1
	# 	max_result = max_result.reset_index().rename({'index': 'Fold'}, axis=1)
	# 	max_result.to_csv(rf'results\homo_mixed_models_{num_estimators}.csv', index=None)

	# 	df_result = pd.concat([df_result, max_result])
	# 	df_result.to_csv(rf'results\homo_mixed_models.csv', index=None)
	# 	print(df_result)







	# result_ = {}
	# max_result = 0

	# for pred_name in models:

	# 	print('Leading ', pred_name)
	# 	result = {}
	# 	models_list = models.copy()
	# 	models_list.remove(pred_name)

	# 	for perm in tqdm(combinations(models_list, 7)):
	# 		models_subset = list(perm)
	# 		models_subset.insert(0, pred_name)

	# 	# for perm in tqdm(permutations(['Lena', 'Dict_247101116', 'German_BERT', 'DiFE_sum_nn', 'OpenSmile_50', 'Boaw_125_50', 'DeepSpectrum'], 7)):
	# 	# for pred_name in preds

	# 	# 	# if 'DenisD' not in list(perm) or 'DenisK' not in list(perm) or 'Lena' not in list(perm) or 'German_BERT' not in list(perm):
	# 	# 	# 	continue

	# 		vote_uars = [] # to store each fold's results
	# 		for i in range(1, 5):

	# 			preds_set = pd.DataFrame({'filename': devel_files[i]})
	# 			for pred_name_ in models_subset:
	# 				preds_set = preds_set.merge(preds[i][pred_name_], on='filename')
				
	# 			preds_set['vote'] = majority_vote(preds_set.drop(['filename'], axis=1))
	# 			preds_set = preds_set.merge(labels, on='filename')
	# 			uar = np.around(recall_score(preds_set['label'], preds_set['vote'], average='macro'), decimals=4)
	# 			vote_uars.append(uar)

	# 	# 		# print(uar)
	# 	# 		# print()
	# 	# 		# for col in preds_set.columns.drop(['filename', 'label']):		
	# 	# 		# 	devel_uar = np.around(recall_score(preds_set['label'], preds_set[col], average='macro'), decimals=4)
	# 	# 		# 	print(col, devel_uar)

			
	# 		name = ', '.join(models_subset)
	# 		result[name] = np.mean(vote_uars)

	# 	# 	if result[name] > max_result:
	# 	# 		max_result = result[name]
	# 	# 		print(max_result, name)
	# 	# 	# print()
	# 	print(pred_name, max(result.values()), max(result, key=result.get))
	# 	print()
	# 	result_[max(result, key=result.get)] = max(result.values())

	# print(max(result_.values()), max(result_, key=result_.get))

			
	# # # # 		# preds.to_csv(os.path.join(ROOT, r'predictions\Devel\CV', f'audio_ensemble_fold_{i}_preds.csv'), index=None)
			

			
	# # # # 	# print(f'\nAverage CV vote UAR: {np.mean(vote_uars)}')

		
