import pandas as pd 
from sklearn.metrics import recall_score
import os
import numpy as np
from collections import Counter

from matplotlib import pyplot as plt

import matplotlib as mpl
import seaborn as sns

import itertools

def plot_lines():

	data = {}
	data['A'] = [78.69, 79.88, 82.01, 82.70, 84.01, 84.40]
	data['L'] = [86.31, 85.84, 87.87, 88.01, 85.84, 85.84]
	data['A+L'] = [86.31, 82.92, 87.42, 93.18, 90.91, 91.99, 90.51]
	data['H(A+L)'] = [86.32, 83.04, 92.00, 88.67, 92.47]
	data['H(A)'] = [79.7775, 81.41, 81.41, 81.41, 81.11, 81.11, 81.07, 79.81, 79.73]

	plt.plot(np.arange(1, len(data['A']) + 1), data['A'], label = "Homogeneous Audio Ensemble")
	plt.plot(np.arange(1, len(data['L']) + 1), data['L'], label = "Homogeneous Lingo Ensemble")
	plt.plot(np.arange(1, len(data['A+L']) + 1), data['A+L'], label = "Homogeneous Mixed Ensemble")
	plt.plot(np.arange(1, len(data['H(A)']) + 1), data['H(A)'], label = "Heterogeneous Audio Ensemble")
	plt.plot(np.arange(1, len(data['H(A+L)']) + 1), data['H(A+L)'], label = "Heterogeneous Mixed Ensemble")


	plt.xlabel('Number of candidates in ensemble')
	plt.ylabel('CV accuracy, UAR (%)')

	plt.legend()

	plt.show()

if __name__ == '__main__':

	root = r'C:\Users\User\Projects\ComParE-2021\results'

	data = {}
	
	with sns.axes_style("white"):

		sns.color_palette("rocket", as_cmap=True)

		data['MAudio'] = pd.read_csv(os.path.join(root, 'homo_audio_models.csv'))
		data['MAudio'] = data['MAudio'].rename({'Estimators' : 'Ensemble size', 'UAR': 'UAR, %'}, axis=1)
		sns.lineplot(data=data['MAudio'], x="Ensemble size", y="UAR, %", label='Audio Ensembles', color='purple')

		data['MLingo'] = pd.read_csv(os.path.join(root, 'homo_lingo_models.csv'))
		data['MLingo'] = data['MLingo'].rename({'Estimators' : 'Ensemble size', 'UAR': 'UAR, %'}, axis=1)
		sns.lineplot(data=data['MLingo'], x="Ensemble size", y="UAR, %", label='Lingo Ensembles', color='orange')

		data['MMixed'] = pd.read_csv(os.path.join(root, 'homo_mixed_models.csv')).iloc[:9*4, :]
		data['MMixed'] = data['MMixed'].rename({'Estimators' : 'Ensemble size', 'UAR': 'UAR, %'}, axis=1)
		sns.lineplot(data=data['MMixed'], x="Ensemble size", y="UAR, %", label='Mixed Ensembles', color='green')

		# data['HAudio'] = pd.read_csv(os.path.join(root, 'hetero_audio_models.csv'))
		# sns.lineplot(data=data['HAudio'], x="Estimators", y="UAR", label='Heterogeneous Audio Ensembles')

		# data['HLingo'] = pd.read_csv(os.path.join(root, 'hetero_lingo_models.csv'))
		# sns.lineplot(data=data['HLingo'], x="Estimators", y="UAR", label='Heterogeneous Lingo Ensembles')

		# data['HMixed'] = pd.read_csv(os.path.join(root, 'hetero_mixed_models.csv'))
		# sns.lineplot(data=data['HMixed'], x="Estimators", y="UAR", label='Heterogeneous Mixed Ensembles')

		plt.legend()

		sns.set_style("whitegrid")
		plt.show()
		