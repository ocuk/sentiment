import pandas as pd
import numpy as np
import os 
from tqdm import tqdm
# from run_sentiment2 import evaluate

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import recall_score, confusion_matrix

from collections import Counter
from itertools import combinations, permutations
from math import factorial

if __name__ == '__main__':

	ROOT = r'C:\Users\User\Projects\ComParE-2021'

	devel_preds = pd.DataFrame(pd.read_csv('y_devel_preds.csv'))
	devel_preds['filename'] = devel_preds['filename'].apply(lambda x: x.split('.')[0])

	preds = [
	'preds_devel_dict_6_best.csv',
	'preds_devel_dict_5_best.csv',
	'preds_devel_transcriptions_Feature_DiFE_sum_nn.csv', 
	# 'preds_devel_transcriptions_Feature_DiFE_sum_sent.csv', 
	# 'preds_devel_transcriptions_Feature_Dutch_FT_delited_stop_words_nonorm.csv', 
	# 'preds_devel_transcriptions_Feature_Dutch_BERT_delited_stop_words.csv',
	# 'preds_devel_transcriptions_Feature_Dutch_BERT_nonorm.csv', 
	# 'preds_devel_transcriptions_Feature_English_BERT_delited_stop_words.csv',
	# 'preds_devel_transcriptions_Feature_English_BERT.csv',
	'preds_devel_transcriptions_Feature_German_BERT_delited_stop_words.csv',
	# 'preds_devel_transcriptions_Feature_German_W2V_100d_delited_stop_words.csv',
	]

	# DD_preds = pd.read_csv(r'C:\Users\User\Projects\ComParE-2021\predictions\preds_devel_MFCC_dynamic_chunks.csv')
	# DD_preds['DD'] = DD_preds.drop(['filename'], axis=1).apply(lambda row: np.argmax(row), axis=1)
	# DD_preds['filename'] = DD_preds['filename'].apply(lambda x: x.split('.')[0])
	# DD_preds = DD_preds[['filename', 'DD']]

	Denis_preds = pd.read_excel(r'C:\Users\User\Projects\ComParE-2021\predictions\Devel\Audio\df_dev_prob_mixup_Denis.xlsx')
	data = np.array(['filename', 'label', 0, 1, 2])
	for i in Denis_preds.iterrows():
		for j in i[1]:
			data = np.vstack([data, j.split(',')])

	Denis_preds = pd.DataFrame(data[1:], columns=data[0])
	Denis_preds['Denis_mixup'] = Denis_preds.drop(['filename', 'label'], axis=1).apply(lambda row: np.argmax(row), axis=1)
	Denis_preds['filename'] = Denis_preds['filename'].apply(lambda x: x.split('.')[0])
	Denis_preds = Denis_preds[['filename', 'Denis_mixup']]
	# Denis_preds['Denis'] = Denis_preds.drop(['filename', 'label'], axis=1).apply(lambda row: np.argmax(row), axis=1)
	# Denis_preds['filename'] = Denis_preds['filename'].apply(lambda x: x.split('.')[0])
	# Denis_preds = Denis_preds[['filename', 'Denis']]

	Alena_preds = pd.read_excel(r'C:\Users\User\Projects\ComParE-2021\predictions\Devel\Audio\preds_devel_Alena')
	data = np.array(['filename', 'label', 0, 1, 2])
	for i in Alena_preds.iterrows():
		for j in i[1]:
			data = np.vstack([data, j.split(',')])

	Alena_preds = pd.DataFrame(data[1:], columns=data[0])
	Alena_preds['Alena'] = Alena_preds.drop(['filename', 'label'], axis=1).apply(lambda row: np.argmax(row), axis=1)
	Alena_preds['filename'] = Alena_preds['filename'].apply(lambda x: x.split('.')[0])
	Alena_preds = Alena_preds[['filename', 'Alena']]


	Lena_preds = pd.read_excel(r'C:\Users\User\Projects\ComParE-2021\predictions\Devel\Audio\df_devel_prob.xlsx')
	# Lena_preds['Denis'] = Denis_preds.drop(['filename', 'label'], axis=1).apply(lambda row: np.argmax(row), axis=1)
	# Lena_preds['filename'] = Denis_preds['filename'].apply(lambda x: x.split('.')[0])
	# Lena_preds = Denis_preds[['filename', 'Denis']]
	
	data = np.array(['filename', 'label', 0, 1, 2])
	for i in Lena_preds.iterrows():
		for j in i[1]:
			data = np.vstack([data, j.split(',')])
	

	Lena_preds = pd.DataFrame(data[1:], columns=data[0])
	Lena_preds['Lena'] = Lena_preds.drop(['filename', 'label'], axis=1).apply(lambda row: np.argmax(row), axis=1)
	Lena_preds['filename'] = Lena_preds['filename'].apply(lambda x: x.split('.')[0])
	Lena_preds = Lena_preds[['filename', 'Lena']]
	Lena_preds.to_csv('preds_devel_audio_Lena.csv', index=None)

	# Read opensmile preds
	op_preds = pd.read_csv(r'C:\Users\User\Projects\ComParE-2021\predictions\Devel\Audio\preds_devel_opensmile_pca250.csv').rename({'0': f'opensmile_pca250'}, axis=1)
	for n in [50, 100, 150, 200]:
		new_df = pd.read_csv(rf'C:\Users\User\Projects\ComParE-2021\predictions\Devel\Audio\preds_devel_opensmile_pca{n}.csv').rename({'0': f'opensmile_pca{n}'}, axis=1)
		op_preds = op_preds.merge(new_df, on='filename')

	# Read boaw preds
	boaw_preds = pd.read_csv(r'C:\Users\User\Projects\ComParE-2021\predictions\Devel\Audio\preds_devel_boaw_125_10.csv').rename({'0': f'boaw_125_10'}, axis=1)
	for book_size in [125, 500, 2000]:
		for noise in [10, 20, 50]:
			if f'boaw_{book_size}_{noise}' in boaw_preds.columns:
				continue
			try:
				new_df = pd.read_csv(rf'C:\Users\User\Projects\ComParE-2021\predictions\Devel\Audio\preds_devel_boaw_125_{noise}.csv').rename({'0': f'boaw_{book_size}_{noise}'}, axis=1)
			except:
				continue
			boaw_preds = boaw_preds.merge(new_df, on='filename')

	dp_preds = pd.read_csv(r'C:\Users\User\Projects\ComParE-2021\predictions\Devel\Audio\preds_devel_deepspectrum.csv').rename({'0': f'deepspectrum'}, axis=1)
	
	# Combine all predefined preds
	for pred_file in preds:

		new_df = pd.read_csv(rf'.\predictions\\Devel\\allpreds\\{pred_file}').rename({'0': pred_file}, axis=1)
		devel_preds = devel_preds.merge(new_df, on='filename')

	devel_preds = devel_preds.rename({'preds': 'Dict'}, axis=1)
	# devel_preds = devel_preds.drop('Dict', axis=1)
	devel_preds = devel_preds.merge(Denis_preds, on='filename')
	# devel_preds = devel_preds.merge(DD_preds, on='filename')
	devel_preds = devel_preds.merge(Lena_preds, on='filename')
	devel_preds = devel_preds.merge(Alena_preds, on='filename')
	devel_preds = devel_preds.merge(op_preds, on='filename')
	devel_preds = devel_preds.merge(boaw_preds, on='filename')
	devel_preds = devel_preds.merge(dp_preds, on='filename')

	# vote = devel_preds.drop(['filename', 'label'], axis=1).astype(int).apply(lambda row: Counter(row).most_common()[0][0], axis=1)
	# devel_uar = np.around(recall_score(devel_preds['label'], vote, average='macro'), decimals=4)

	# for col in devel_preds.columns.drop(['filename', 'label']):
		
	# 	devel_uar = np.around(recall_score(devel_preds['label'], devel_preds[col], average='macro'), decimals=4)
	# 	print(col, devel_uar)

	result = {}
	
	preds = [
	'Dict',
	'preds_devel_dict_5_best.csv',
	# 'preds_devel_dict_6_best.csv', 
	'preds_devel_transcriptions_Feature_DiFE_sum_nn.csv', 
	# 'preds_devel_transcriptions_Feature_DiFE_sum_sent.csv', 
	# 'preds_devel_transcriptions_Feature_Dutch_FT_delited_stop_words_nonorm.csv', 
	# 'preds_devel_transcriptions_Feature_Dutch_BERT_delited_stop_words.csv',
	# 'preds_devel_transcriptions_Feature_Dutch_BERT_nonorm.csv', 
	# 'preds_devel_transcriptions_Feature_English_BERT_delited_stop_words.csv',
	# 'preds_devel_transcriptions_Feature_English_BERT.csv',
	'preds_devel_transcriptions_Feature_German_BERT_delited_stop_words.csv',
	# 'preds_devel_transcriptions_Feature_German_W2V_100d_delited_stop_words.csv',
	'Denis_mixup',
	'deepspectrum',
	'Lena',
	'Alena',
	'opensmile_pca250',
	'opensmile_pca50',
	'boaw_125_50',
	# 'DD',
	
	]

	tmp = devel_preds[preds]
	vote = tmp.astype(int).apply(lambda row: Counter(row).most_common()[0][0], axis=1)
	devel_preds['vote'] = vote

	tmp = devel_preds[['filename', 'label', 'vote'] + preds]
	# tmp = tmp.rename({'preds_devel_dict_6_best.csv': 'dict_6_best', 
	# 	'preds_devel_transcriptions_Feature_DiFE_sum_nn.csv': 'DiFE_sum_nn', 
	# 	'preds_devel_transcriptions_Feature_German_BERT_delited_stop_words.csv': 'German_BERT_delited_stop_words',
	# 	}, axis=1)
	# tmp.to_csv('best_preds_for_audio_voting.csv', index=None)

	for col in tmp.columns.drop(['filename', 'label']):		
		devel_uar = np.around(recall_score(devel_preds['label'], devel_preds[col], average='macro'), decimals=4)
		print(col, devel_uar)

	# # # devel_uar = np.around(recall_score(tmp['label'], tmp['vote'], average='macro'), decimals=4)
	# # # print(devel_uar)

	# # # conf_m = confusion_matrix(devel_preds['label'], vote)
	# # # print(devel_preds[devel_preds['label'] != devel_preds['vote']])

	# # # print()
	
	# # # num_cols = len(devel_preds.drop(['filename', 'label'], axis=1).columns)
	num_cols = len(preds)
	num_preds = 6
	print(f'\nSearching {factorial(num_cols) / (factorial(num_preds) * factorial(num_cols - num_preds))} combinations')

	# # # combs = combinations(devel_preds.drop(['filename', 'label'], axis=1).columns, num_preds)
	# # # for comb in tqdm(combs):
	for comb in tqdm(combinations(preds, num_preds)):

		tmp = devel_preds[list(comb)]
		vote = tmp.astype(int).apply(lambda row: Counter(row).most_common()[0][0], axis=1)
		devel_uar = np.around(recall_score(devel_preds['label'], vote, average='macro'), decimals=4)
		name = ', '.join(map(lambda x: str(x), list(comb)))
		result[name] = devel_uar

	print(max(result.values()), max(result, key=result.get))
	print()


	# # # for k,v in reversed(sorted(result.items(), key=lambda p:p[1])):
	# # # 	print(v, k)
		
		
