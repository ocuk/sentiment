import pandas as pd
import numpy as np
import re
import os

import json
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer

import pickle

from tqdm import tqdm

import stats


class SentiWordNet:

	def __init__(self, path):

		path = os.path.join(path, 'SentiWordNet.csv')
		senti_dict = pd.read_csv(path, dtype={'ID': int})
		self.dictionary = senti_dict

	def map_pos(self, pos):

		if pos == 'j':
			return 'a'
		else:
			return pos

	def look_up(self, word, score_type, pos=None, flip=False, debug=False):
		'''
		Checks the dictinary for a given word and returns the mean score 
		for a given score type (positive or negative) 
		
		Arguments:
			word -  
			score_type - PosScore or NegScore 
			pos - part of speech tag
			flip - if True, the score is reversed 
			debug - see below
		
		Returns:
			if debug=True: a tuple of type (word, pos, score) or None
			if debug=False: a float (mean of sentiment scores for a given word)
		'''

		# Find the right token
		score_list = self.dictionary[self.dictionary['Word'] == word]

		# Find the right Part of Speech
		if pos:
			score_list = score_list[score_list['POS']==pos]

		# Create a list of scores (positive or negative)
		score_list = score_list[score_type].to_list()

		# If a sentiment is found, return the mean score
		if not all(v == 0 for v in score_list):
			score = np.round(np.mean(score_list), 5)
			
			# if the word is preceded with a negation:
			if flip:
				score = -score
				token = flip + ' ' + word
			else:
				token = word
			
			if debug:
				return (token, pos, score)
			else:
				return score
		
		# Otherwise return None
		else:
			return None

	def get_scores(self, text, score_type, pos=True, keep_dim=True, debug=False):
		'''
		Checks the dictionary for every word in text and returns 
		the scores of type (word, pos, score)

		Arguments:
			text - a string with original text
			score_type - a string of score type (PosScore or NegScore)
			keep_dim - whether to include non-sentiment tokens in the list to
				maintain the original length of the sequence. The scores associated with
				non-sentiment tokens are included as None 

		Returns: 
			if debug=True: a list of tuples of type (word, pos, score) or None
			if debug=False: a list of scores 
		'''

		txt = text.lower()
		txt = txt.replace(r'&quot;', r'"')
		tokens = nltk.word_tokenize(txt)
		
		if pos:
			tokens = nltk.pos_tag(tokens)

		lemmatizer = WordNetLemmatizer()

		scores = []
		for token in tokens:
			if pos:
				pos_tag = token[1][0].lower()
				pos_tag = self.map_pos(pos_tag)
				token = token[0]

			# Applies if negation is True and some tokens contain two word (e.g. "not good")
			flip = False
			if len(token.split(' ')) == 2 and token.startswith('no'):
				flip = token.split(' ')[0]
				token = token.split(' ')[1]
			if not pos:
				pos_tag = None

			# Check token in the dictionary
			token_score = self.look_up(token, score_type, pos=pos_tag, flip=flip, debug=debug)

			# If token is found, add the score
			if token_score:
				scores.append(token_score)

			# If token is not found, check the lemmata
			else:
				# Try to find the lemmata
				try:
					lemmata = lemmatizer.lemmatize(token, pos=pos)
					lemmata_score = look_up(lemmata, score_type, pos=pos_tag, flip=flip, debug=debug)
					
					# If keep_dim=True, add whatever the lemmata score is found (score or None)
					if keep_dim:
						scores.append(lemmata_score)
					
					# If keep_dim=False and the lemmata is found in the dictionary, add the score
					elif lemmata_score is not None:
						scores.append(lemmata_score)
				
				# If no lemmata is found, and keep_dim=True, add the token score (None) 
				except:
					if keep_dim:
						scores.append(token_score)				
		return scores

	def get_stats(self, text, score_type, pos=True, keep_dim=False, selected=False):
		'''
		Calculates the sentiment scores of the text and returns statistics

		Arguments:
			text - original string of text
			score_type - a string of score type (PosScore or NegScore)
			pos - bool to indicate whether to consider part of speech tags
			keep_dim - should always be False

		Returns: 
			a dictionary with min, max, mean, sum, num and range statistics

		'''

		assert keep_dim == False, 'keep_dim should be False when calculating statistics'
		score_list = self.get_scores(text, score_type, pos=pos, keep_dim=False)

		if not score_list:
			score_list = [0]

		stats = {}
		stats['min'] = np.min(score_list)
		stats['max'] = np.max(score_list)
		stats['mean'] = np.mean(score_list)
		stats['range'] = stats['max'] - stats['min']
		stats['sum'] = np.sum(score_list)
		stats['num'] = len(score_list)
			
		if selected:
			selected_stats = {}
			
			for stat in selected:
				selected_stats[stat] = stats[stat]

			return selected_stats
		return stats
	

class SentiWS:

	def __init__(self, path, pos):

		path = os.path.join(path, 'SentiWS.csv')

		self.dictionary = pd.read_csv(path)
		self.dictionary['Base'] = self.dictionary['Base'].apply(self.make_lowercase)
		self.stemmer = SnowballStemmer('german')

		if pos:
			with open('nltk_german_classifier_data.pickle', 'rb') as f:
				self.tagger = pickle.load(f)
		else:
			self.tagger = None

		self.dictionary_stemmed = self.dictionary.copy()
		self.dictionary_stemmed['Base'] = self.dictionary_stemmed['Base'].apply(self.stemmer.stem)

	def check_inflections(self, inflection_list, word):

		if inflection_list and word in inflection_list:
			return True
		else:
			return False

	def make_lowercase(self, word):
		return word.lower()


	def look_up(self, word, pos=None, flip=False, debug=False, use_stemmer=False):
		'''
		Checks the dictinary for a given word and returns the mean score 

		Arguments:
			word - 
			pos - 
			flip - 
			debug - 
			stemmer - 

		Returns:
			if debug=True: a tuple of type (word, pos, score) or None
			if debug=False: a float (mean of sentiment scores for a given word)

		'''

		# Find the right token
		score_list = self.dictionary[self.dictionary['Base'] == word]

		# Check inflections
		if score_list.empty:
			mask = self.dictionary['Inflections'].apply(self.check_inflections, args=(word,))
			score_list = self.dictionary[mask]

			# Check stemmed words
			if score_list.empty and use_stemmer:
				word = self.stemmer.stem(word)
				score_list = self.dictionary_stemmed[self.dictionary_stemmed['Base'] == word]

		# Find the right Part of Speech
		if pos:
			score_list = score_list[score_list['POS']==pos]

		# Create a list of scores
		score_list = score_list['Score'].to_list()

		# If a sentiment is found, return the mean score
		if not all(v == 0 for v in score_list):
			score = np.round(np.mean(score_list), 5)

			# if the word is preceded with a negation, add a negation
			# word before the token
			if flip:
				score = -score
				token = flip + ' ' + word
			else:
				token = word

			if debug:
				return (token, pos, score)
			else:
				return score


	def get_scores(self, text, pos=False, keep_dim=False, debug=False, use_stemmer=False):
		'''
		For the given text provide a word-based list of scores found in the SentiWS dictionary
		'''

		txt = text.lower()
		txt = txt.replace(r'&quot;', r'"')
		tokens = nltk.word_tokenize(txt)

		if pos:
			tokens = self.tagger.tag(tokens)

		scores = []
		for token in tokens:
			if pos:
				pos_tag = token[1][0].lower()
				token = token[0]

			# Applies if negation is True and some tokens contain two word (e.g. "not good")
			flip = False
			if len(token.split(' ')) == 2 and token.startswith('no'):
				flip = token.split(' ')[0]
				token = token.split(' ')[1]
			if not pos:
				pos_tag = None

			# Check token in the dictionary
			token_score = self.look_up(token, pos=pos_tag, flip=flip, debug=debug, use_stemmer=use_stemmer)

			# If token is found, add the score
			if keep_dim:
				scores.append(token_score)
			elif token_score is not None:
				scores.append(token_score)

		return scores


	def get_stats(self, text, pos=False, keep_dim=False, use_stemmer=False, selected=False):
		'''
		return a dictionary with score statistics corresponding to the input sentence.
		'''

		assert keep_dim == False, 'keep_dim should be False when calculating statistics'
		score_list = self.get_scores(text, pos=pos, keep_dim=False, use_stemmer=use_stemmer)
		
		if not score_list:
			score_list = [0]

		stats = {}
		stats['min'] = np.min(score_list)
		stats['max'] = np.max(score_list)
		stats['mean'] = np.mean(score_list)
		stats['range'] = stats['max'] - stats['min']
		stats['sum'] = np.sum(score_list)

		mask = np.greater(score_list, 0)
		stats['num_pos'] = np.array(score_list)[mask].size

		mask = np.less(score_list, 0)
		stats['num_neg'] = np.array(score_list)[mask].size

		if selected:
			selected_stats = {}
			
			for stat in selected:
				selected_stats[stat] = stats[stat]

			return selected_stats
		return stats


if __name__ == '__main__':

	ROOT = r'C:\Users\User\Projects\ComParE2020_Elderly'
	dict_path = os.path.join(ROOT, 'dictionaries', 'SentiWS')
	d = SentiWS(dict_path)
	print(d.get_stats('Die Katze liegt auf der Matte.', pos=False))
	print(d.get_stats('Die Katze liegt auf der Matte.', pos=False, selected=['min', 'max']))
	
