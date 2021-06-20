import pandas as pd 
from sklearn.metrics import recall_score
import os
import numpy as np
from collections import Counter

from sklearn.metrics import recall_score
from matplotlib import pyplot as plt

def majority_vote(df):

	vote = df.astype(int).apply(lambda row: Counter(row).most_common()[0][0], axis=1)
	return vote


def majority_vote_with_rule_2(df):

	def apply_rule(row):

		val1 = row[0]
		val2 = row[1]

		if val1 == val2:
			return val1
		else:
			return 1

	vote = df.astype(int).apply(lambda row: apply_rule(row), axis=1)
	return vote


def majority_vote_with_rule(df):

	def apply_rule(row):

		c = Counter(row)
		max_count = c.most_common()[0][1]

		polemic_classes = []
		for key in c:
			if c[key] == max_count:
				polemic_classes.append(key)

		if len(polemic_classes) == 1:
			return polemic_classes[0]
		else:
			print(polemic_classes, end=' -> ')
			if 0 in polemic_classes and 2 in polemic_classes:
				print('Chose 1')
				return 1
			else:
				print('Chose 2')
				return 2

	vote = df.astype(int).apply(lambda row: apply_rule(row), axis=1)
	return vote


def majority_vote_with_rule_Nastya(df):

	def apply_rule(row):

		c = Counter(row)
		max_count = c.most_common()[0][1]

		polemic_classes = []
		for key in c:
			if c[key] == max_count:
				polemic_classes.append(key)

		if len(polemic_classes) == 1:
			return polemic_classes[0]
		else:

			print(polemic_classes, end=' -> ')
			if 0 in polemic_classes:
				if 2 in polemic_classes:
					print('Chose 1')
					return 1
				else:
					print('Chose 0')
					return 0
			else:
				print('Chose 1')
				return 1  


	vote = df.astype(int).apply(lambda row: apply_rule(row), axis=1)
	return vote
if __name__ == '__main__':

	ROOT = r'C:\Users\User\Projects\ComParE-2021'

	test_preds = pd.DataFrame(columns=['Fold1', 'Fold2', 'Fold3', 'Fold4'])

	for i in range(1, 5):
		print('Fold ', i)
		# preds = pd.read_csv(os.path.join(ROOT, rf'predictions\Test\DeepSpectrum\deepspectrum_test_preds_fold_{i}.csv'))
		# preds = preds.rename({'dp_preds': 'DeepSpectrum'}, axis=1)

		if i == 1:
			model_names = ['DenisK', 'Boaw_125_50']
		elif i == 2:
			model_names = ['OpenSmile_50', 'DenisK', 'Boaw_125_50', 'Boaw_2000_50']
		elif i == 3:
			model_names = ['DeepSpectrum', 'Lena', 'DenisD', 'OpenSmile_50', 'Boaw_125_50']
		elif i == 4:
			model_names = ['DeepSpectrum', 'Lena', 'DenisD', 'Boaw_125_50']

		preds = pd.read_csv('test_labels.csv').drop('label', axis=1)
		preds['filename'] = preds['filename']

		for pred_name in model_names:
			print('Readin')
			try:
				df = pd.read_csv(os.path.join(ROOT, rf'predictions\\Test\\{pred_name}\\df_test_prob_{i}.csv'))

				if 'label' in df.columns:
					df[pred_name] = df.drop(['filename', 'label'], axis=1).apply(lambda row: np.argmax(row), axis=1)
				else:
					df[pred_name] = df.drop(['filename'], axis=1).apply(lambda row: np.argmax(row), axis=1)
				df = df[['filename', pred_name]]
			except:
				try:
					df = pd.read_csv(os.path.join(ROOT, rf'predictions\Test\\{pred_name}\\df_test_pred_{i}.csv'))
					tmp_col_name = list(df.columns)
					tmp_col_name.remove('filename')
					df = df[['filename', tmp_col_name[0]]]
					df.columns = ['filename', pred_name]
				except:
					print('Couldn\'t read', pred_name)
			
			df['filename'] = df['filename'].apply(lambda x: x if x.endswith('.wav') else x + '.wav')

			preds = preds.merge(df, on='filename')
		
		preds['vote'] = preds.drop('filename', axis=1).astype(int).apply(lambda row: Counter(row).most_common()[0][0], axis=1)
		test_preds[f'Fold{i}'] = preds['vote']

	test_preds['Vote'] = test_preds.astype(int).apply(lambda row: Counter(row).most_common()[0][0], axis=1)
	test_preds['filename'] = preds['filename']

	# test_preds = test_preds[['filename', 'Fold1', 'Fold2', 'Fold3', 'Fold4', 'Vote']]
	test_preds = test_preds[['filename', 'Vote']]
	
	test_preds = test_preds.rename({'Vote': 'prediction'}, axis=1)
	test_preds['filename'] = test_preds['filename']
	
	print(test_preds)
	test_preds.to_csv('TeamAlexeyKarpov_ESS_trial_4.csv', index=None)
