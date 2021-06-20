import pandas as pd 
from sklearn.metrics import recall_score
import os
import numpy as np
from collections import Counter

from matplotlib import pyplot as plt

if __name__ == '__main__':

	ROOT = r'C:\Users\User\Projects\ComParE-2021'
	
	df = pd.read_csv(r'TeamAlexeyKarpov_ESS_trial_4.csv')
	f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)

	labels = list(df['prediction'].value_counts().index)
	counts = df['prediction'].value_counts().unique()
	data = df['prediction'].value_counts()
	patches, texts = ax1.pie(data, labels=counts)
	ax1.set_title('Mixed heterogeneous ensemble')
	ax1.legend(patches, labels, loc="best")

	df = pd.read_csv(r'TeamAlexeyKarpov_ESS_trial_3.csv')

	labels = list(df['prediction'].value_counts().index)
	counts = df['prediction'].value_counts().unique()
	data = df['prediction'].value_counts()
	patches, texts = ax2.pie(data, labels=counts)
	ax2.set_title('Mixed ensemble')
	ax2.legend(patches, labels, loc="best")

	df = pd.read_csv(r'TeamAlexeyKarpov_ESS_trial_1.csv')

	labels = list(df['prediction'].value_counts().index)
	counts = df['prediction'].value_counts().unique()
	data = df['prediction'].value_counts()
	patches, texts = ax3.pie(data, labels=counts)
	ax3.set_title('Acoustic ensemble')
	ax3.legend(patches, labels, loc="best")

	
	df = pd.read_csv(r'TeamAlexeyKarpov_ESS_trial_2.csv')
	
	labels = list(df['prediction'].value_counts().index)
	counts = df['prediction'].value_counts().unique()
	data = df['prediction'].value_counts()
	patches, texts = ax4.pie(data, labels=counts)
	ax4.set_title('Linguistic ensemble')
	ax4.legend(patches, labels, loc="best")

	plt.suptitle('Test set predictions')
	plt.show()

	# df = df.merge(test_preds, on='filename')
	# print(df[df['prediction_x'] == df['prediction_y']])
	# print(df)
