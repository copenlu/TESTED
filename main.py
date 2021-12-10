import argparse
import pickle
import json
from typing import List
from weak_labeler import WeakLabeler
import os

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
	'--parallelize',type=bool,default=True,dest='parallelize',
	help="A flag for using multiprocessing"
	)

	parser.add_argument("--dont-parallelize",dest='parallelize', action="store_false")

	args = parser.parse_args()

	annotator = WeakLabeler(model_type = args.model_name, labeler_type = args.labeler_type, \
	dataset_path = args.dataset_path, config_path = args.config_path, parallelize = args.parallelize)
	annotator.weak_labeler_parallel()
