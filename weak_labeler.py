from multiprocessing.pool import ThreadPool
from transformers import AutoTokenizer, AutoModelForMaskedLM,AutoModelForSequenceClassification
from utils import list_segmentor, insert_scheme
from env_loader import WeakLabelerSingelton
import torch.multiprocessing as mp
from collections import defaultdict
from transformers import pipeline
from typing import Dict, List
from tqdm import tqdm
import multiprocess
import asyncio
import torch
import json
import time
import os

from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne
from pprint import pprint

from evaluation_mappings import TASK_MAPPINGS


class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

class WeakLabeler:
	'''
	 Apparently because multiprocessing copies((de)serialize) instace attributes for each process, it is unable to use the generator and the model
	'''

	env_singelton = None
	labeler_metadata = None

	def __init__(self, model_type: str = '', labeler_type: str = 'zero', dataset_path: str = '', config_path: str = '', parallelize: bool = True, cuda_index: int = 0):

		env = WeakLabelerSingelton.getInstance()

		# Currently we use only one gpu, bt the same idea works for several indices
		device = cuda_index if 'cuda' in WeakLabelerSingelton.cuda.type else -1

		WeakLabeler.labeler_metadata = {}

		WeakLabeler.labeler_metadata['device'] = device
		WeakLabeler.labeler_metadata['model_type'] = model_type
		WeakLabeler.labeler_metadata['labeler_type'] = labeler_type
		WeakLabeler.labeler_metadata['data_path'] = dataset_path
		WeakLabeler.labeler_metadata['model_type'] = model_type
		WeakLabeler.labeler_metadata['prompts'] = self.read_config(config_path) if 'mlm' in labeler_type else None
		WeakLabeler.labeler_metadata['zero'] = self.read_config(config_path) if 'zero' in labeler_type else None
		WeakLabeler.labeler_metadata['parallelize'] = parallelize

		WeakLabeler.labeler_metadata = dotdict(WeakLabeler.labeler_metadata)


		self.insertion_scheme = WeakLabeler.labeler_metadata.prompts['insertion_scheme'] if WeakLabeler.labeler_metadata.prompts is not None \
			else  WeakLabeler.labeler_metadata.zero['insertion_scheme']

		# self.model_type = model_type
		# self.labeler_type = labeler_type
		#
		# self.data_path = dataset_path
		#
		# self.prompts = self.read_config(config_path) if 'mlm' in self.labeler_type else None
		# self.zero = self.read_config(config_path) if 'zero' in self.labeler_type else None
		# self.parallelize = parallelize

		if env.model is  None:

			print(f"Loading the model: {model_type}")

			if 'generator' in WeakLabeler.labeler_metadata.labeler_type:

				# The Idea was scraped because the pipeline is the slowest and big LMs are needed
				labeler =  pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', device = WeakLabeler.labeler_metadata.device)
				tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')

			elif 'mlm' in WeakLabeler.labeler_metadata.labeler_type :
				model = AutoModelForMaskedLM.from_pretrained(model_type)

				if WeakLabeler.labeler_metadata.parallelize:
					model.share_memory()

				tokenizer = AutoTokenizer.from_pretrained(model_type)
				labeler = pipeline('fill-mask', model = model, tokenizer=tokenizer, framework="pt", device = WeakLabeler.labeler_metadata.device)

			elif 'zero' in WeakLabeler.labeler_metadata.labeler_type:
				model = AutoModelForSequenceClassification.from_pretrained(model_type)
				if WeakLabeler.labeler_metadata.parallelize:
					model.share_memory()

				tokenizer = AutoTokenizer.from_pretrained(model_type)
				labeler = pipeline('zero-shot-classification', model = model, tokenizer=tokenizer, framework="pt", device = WeakLabeler.labeler_metadata.device)


			attributes = dict(model=model, tokenizer=tokenizer, pipeline=labeler, labeler_metadata=WeakLabeler.labeler_metadata, insertion_scheme=self.insertion_scheme)
			env.set_attr(**attributes)

			print("Loading completed")

	##################
	###Config Loaders###
	##################


	def read_config(self, config_path: str = 'configs/populated_prompts.json') -> Dict:
		configs = None
		try:
			with open(config_path, 'r') as openfile:
				configs = json.load(openfile)

		except RuntimeError as e:
			print('Cannot read the prompts with error: ',e)
			return configs
		return configs

	##################
	## Processing   ##
	##################

	def mongo_bulk_write(self, queries: List = [], updates: List = []) :
		bulk_result = None
		try:
			env = WeakLabelerSingelton.getInstance()

			requests = [UpdateOne(queries[i], updates[i], upsert=True) for i in range(len(updates))]
			bulk_result = env.collection.bulk_write(requests)

		except RuntimeError as e:
			print("Cannot bulk write with error: ", e)
			return bulk_result
		return bulk_result

	def post_porcess_mlm(self, results: List = []) -> List:
		results_processed = []
		try:

			for result in results:
				if '_' in result[0]['token_str'][0]:
					result_processed = result[0]['token_str'][1:]
				else:
					result_processed = result[0]['token_str']

				results_processed.append(result_processed)


		except RuntimeError as e:
			print('Cannot post process MLM results with the error: ', e)
			return results_processed
		return results_processed

	def post_porcess_zero(self, results: List = [], results_sentiment: List = []) -> List:
		results_processed = []
		try:
			for index, result in enumerate(results):
				results_processed.append([result, results_sentiment[index]])

		except RuntimeError as e:
			print('Cannot post process Zero-shot results with the error: ', e)
			return results_processed
		return results_processed

	def process_tweets(self, user_tweets, labeler_type, languages) -> List:
		processed_tweets = None
		try:
			if 'zero' in labeler_type.lower():
				processed_tweets = user_tweets
			elif 'mlm' in labeler_type.lower():
				processed_tweets = self.process_tweets_mlm(user_tweets, languages)
			elif 'generator' in labeler_type.lower():
				processed_tweets = self.process_tweets_generator(user_tweets, languages)

		except RuntimeError as e:
			print(e)
			return processed_tweets
		return processed_tweets

	def process_tweets_mlm(self, user_tweets: List = [], language: str = '') -> List:
		processed_tweets = []
		try:
			prompts = WeakLabeler.labeler_metadata.prompts
			prompt_meta_dict = prompts[language]

			for user_meta in user_tweets:
				for key in prompt_meta_dict:
					processed_meta = {}

					prompt, pos_lab, neg_lab = prompt_meta_dict[key]
					process_meta = user_meta.copy()
					process_meta['full_text'] = f'{user_meta["full_text"]} {prompt}'
					process_meta['class'] = key

					process_meta['pos_lab'] = pos_lab
					process_meta['neg_lab'] = neg_lab
					process_meta['prompt'] = prompt


					processed_tweets.append(process_meta)

		except RuntimeError as e:
			print('Cannot process for the MLM labeler with error: ',e)
			return processed_tweets
		return processed_tweets

	def process_tweets_zero(self, user_tweets: List = [], language: str = '') -> List:
		processed_tweets = []
		try:
			prompts = self.read_config(config_path='configs/populated_zero.json')
			zero_meta_dict = prompts[language]

			for user_meta in user_tweets:
				processed_meta = {}
				topics, sentiments, hypothesis_template = zero_meta_dict['Topics'],\
															zero_meta_dict['Sentiments'], \
																zero_meta_dict['Hypothesis']


		except RuntimeError as e:
			print('Cannot process for the Zero-shot labeler with error: ',e)
			return processed_tweets
		return processed_tweets

	def process_tweets_generator(self, user_tweets, language):
		processed_tweets = None
		try:
			primes = None
		except RuntimeError as e:
			print(e)
			return processed_tweets
		return processed_tweets

	##################
	##    Labeler   ##
	##################

	def weak_labeler_instance(self, user_metas: List = [] ):
		bulk_result = None
		results = defaultdict(lambda: defaultdict(dict))
		try:
			env = WeakLabelerSingelton.getInstance()

			tweets_lang = user_metas[0]['lang']

			user_metas = self.process_tweets(user_metas, env.labeler_metadata.labeler_type, tweets_lang)

			user_ids = [user_meta['user_id'] for user_meta in user_metas]
			tweet_ids = [user_meta['tweet_id'] for user_meta in user_metas]
			tweet_texts = [user_meta['full_text'] for user_meta in user_metas]

			if 'label' in user_metas[0]:
				labels = [user_meta['label'] for user_meta in user_metas]
			else:
				labels = None

			
			if 'mlm' in env.labeler_metadata.labeler_type.lower():

				pos_lab = user_metas[0]['pos_lab']
				neg_lab = user_metas[0]['neg_lab']


				# Numworkers doesnt really matter as long as threadcout is similar
				results_raw =  env.pipeline(tweet_texts, targets = [pos_lab, neg_lab], num_workers = 0)
				results_processed = self.post_porcess_mlm(results_raw)

				assert(len(user_metas) == len(results_processed))

			elif 'zero' in env.labeler_metadata.labeler_type.lower():

				zero_meta_dict = env.labeler_metadata.zero
				zero_meta_dict = zero_meta_dict[tweets_lang]
				topics, sentiments, hypothesis_template = zero_meta_dict['Topics'],\
															zero_meta_dict['Sentiments'], \
																zero_meta_dict['Hypothesis']


				results_raw = env.pipeline(tweet_texts, topics, hypothesis_template=hypothesis_template, num_workers = 0)
				results_sentiment = env.pipeline(tweet_texts, sentiments,hypothesis_template = hypothesis_template, num_workers = 0)

				results_processed = self.post_porcess_zero(results_raw, results_sentiment)

				assert(len(user_metas) == len(results_processed))


			elif 'generator' in env.labeler_metadata.labeler_type.lower():

				# TODO: Generator pipeline was removed becase it was slow. Maybe better GPU's ?
				annotated_tweets, primes, labels = self.process_tweets(tweet_texts)
				tweet_tokens = env.tokenizer(tweet_texts)
				tweet_tokens_len = max([len(tok) for tok in tweet_tokens])
				results_texts = self.labeler(annotated_tweets, do_sample=True, min_length = 1, max_length = 1)


			for index, user_meta in enumerate(user_metas):

				user_id = user_ids[index]
				tweet_id = tweet_ids[index]
				label = labels[index] if labels is not None else None

				results[user_id][tweet_id]['label'] = label
				results[user_id][tweet_id]['results_processed'] = results_processed[index]
				results[user_id][tweet_id]['labeler_type'] = WeakLabeler.labeler_metadata.labeler_type

				if 'mlm' in env.labeler_metadata.labeler_type:
					if user_meta['class'] not in results[user_id][tweet_id]:
						per_class_processor = {}
						per_class_processor['class'] = user_meta['class']
						per_class_processor['result_text'] = results_raw[index]
						per_class_processor['pos_lab'] = pos_lab
						per_class_processor['neg_lab'] = neg_lab
						results[user_id][tweet_id][f'result_for_class_{user_meta["class"]}'] = per_class_processor

						# results[user_id][tweet_id]['result_text'] = results_raw[index]
						# results[user_id][tweet_id]['class'] = user_meta['class']
						# results[user_id][tweet_id]['pos_lab'] = pos_lab
						# results[user_id][tweet_id]['neg_lab'] = neg_lab

			queries = [{"user_id": f"{user_id}"} for user_id in user_ids]
			updates = [{f"{tweet_id}.{env.labeler_metadata.labeler_type.lower()}":results[user_ids[index]][tweet_id]} for index, tweet_id in enumerate(tweet_ids)]
			updates = [{"$set":update} for update in updates]

			bulk_result = self.mongo_bulk_write(queries, updates)

		except RuntimeError as e:
			print('Cannot process the instance with error: ', e)
			return bulk_result
		return bulk_result


def weak_labeler_parallel(weak_labeler: WeakLabeler,num_workers = 8) -> Dict:
	results = defaultdict(lambda: defaultdict(dict))
	env = WeakLabelerSingelton.getInstance()

	try:
		accumulator = []

		#A Very hacky solution: make the forker believe we are monlith to avoid deadlocks. Can change the number after forking.
		torch.set_num_threads(1)
		os.environ["TOKENIZERS_PARALLELISM"] = "false"

		# Arguably base multiprocess > torch.multiprocess , which is weird
		if weak_labeler.labeler_metadata.device >= 0:
			torch.backends.cudnn.enabled = True
			mp.set_start_method('spawn',force=True)
		else:
			mp.set_start_method('fork',force=True)


		# mp.set_sharing_strategy('file_system')
		porcesses = ThreadPool(num_workers)
		
		#reset to default after forking
		torch.set_num_threads(num_workers)

		for ind, user_metas in tqdm(enumerate(env.data_loader.data_generator())):
			if len(accumulator) < num_workers:
				accumulator.append(user_metas.copy())
			else:
				# async mapping is better/faster but should write a callback for compeltion
				results = porcesses.map(weak_labeler.weak_labeler_instance, accumulator)
				accumulator = []
				accumulator.append(user_metas)
				# porcesses = mp.Pool(num_workers)
			

		#reset to default thread count
		os.environ["TOKENIZERS_PARALLELISM"] = "true"

		for el in accumulator:
			weak_labeler.weak_labeler_instance(el)

	except RuntimeError as e:
		print('Cannot label with error: ', e)
		return results
	return results




if __name__ == '__main__':
	# Supress TF native output
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

	# annotator = WeakLabeler(model_type='xlm-roberta-base', labeler_type='mlm',
    #                      dataset_path='Data/example_~100_variation.json', config_path='configs/populated_prompts.json', parallelize=True)

	annotator = WeakLabeler(model_type = 'vicgalle/xlm-roberta-large-xnli-anli', labeler_type = 'zero', \
	dataset_path = 'Data/example_~100_variation.json', config_path = 'configs/populated_zero.json', parallelize = True)

	annotator.weak_labeler_parallel()