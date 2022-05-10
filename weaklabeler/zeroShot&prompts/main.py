import os
import argparse
import pickle
import json
from typing import List
from data_loader import DataLoader
from env_loader import WeakLabelerSingelton
from weak_labeler import WeakLabeler, weak_labeler_parallel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

if __name__ == "__main__":

	labeler_types = ['zero', 'mlm']

	parser = argparse.ArgumentParser(
	description="Weak Labeler Namespace"
	)

	parser.add_argument(
	'--model_name',
	help="The name of the LM. Loading from huggingface"
	)

	parser.add_argument(
	'--config_path',type=str,
	help="The path to the config for preprocessing"
	)

	parser.add_argument(
		'--dataset_path',type=str,
		help="The path to the dataset for labeling (Note it is gone to be streamed)"
		)

	parser.add_argument(
	'--labeler_type',type=str,choices = labeler_types,
	help="The type of the labeling method to use"
	)
	parser.add_argument(
	'--hypothesis_separator',type=str,default=None,
	help="The separator mode to use for zero shot"
	)

	parser.add_argument(
	'--mongo_db',type=str,default='Users',
	help="The name of the MongoDB database"
	)

	parser.add_argument(
	'--mongo_collection',type=str,default='Processed_Tweets',
	help="The name of the MongoDB collection"
	)

	parser.add_argument(
	'--mongo_client',type=str,default='mongodb://localhost:27017/',
	help="The name of the MongoDB client (ip:port)"
	)

	parser.add_argument(
	'--parallelize',type=bool,default=True,dest='parallelize',
	help="A flag for using multiprocessing"
	)

	parser.add_argument("--dont-parallelize",dest='parallelize', action="store_false")

	args = parser.parse_args()

	mongo_client, mongo_db, mongo_collection  = args.mongo_client, args.mongo_db, args.mongo_collection


	env = WeakLabelerSingelton.getInstance()

	data_loader = DataLoader(args.dataset_path)
	annotator = WeakLabeler(model_type = args.model_name, labeler_type = args.labeler_type,\
		 config_path = args.config_path, parallelize = args.parallelize, hypothesis_separator = args.hypothesis_separator)


	attribures = dict( mongo_client = mongo_client, mongodb = mongo_db, collection = mongo_collection, data_loader = data_loader)
	print(attribures)
	env.set_attr(**attribures)


	weak_labeler_parallel(annotator)
