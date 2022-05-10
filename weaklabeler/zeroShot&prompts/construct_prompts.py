import argparse
import pickle
import json

from googletrans import Translator
import translators as ts

from typing import Dict, List
from tqdm import tqdm


#This preprocessing can be done better
def preprocess_prompts(text: str = '', mask_token: str = '<mask>') -> str:
	processed_text = None
	try:
		mask_enclosing = mask_token[0] + mask_token[-1]
		processed_text = text.replace(mask_token, mask_enclosing)
	except RuntimeError as e:
		print("Cannot process prompt with error: ", e)
		return processed_text
	return processed_text


def translate(prompt:str = '', lang: str = '') -> str:
	prompt_trans = None
	try:
		# make sure this is cached not reloaded every time
		translator = Translator()

		# print(lang)
		prompt_trans = translator.translate(prompt, dest=lang)
	except ValueError as e:
		print("Cannot translate prompt with error: ", e)
		return prompt_trans
	return prompt_trans.text


def populate_prompts(languages: List = [], base_prompts: Dict = {}, mask_token: str = '<mask>') -> Dict:
	prompts = {}
	try:

		mask_enclosing = mask_token[0]+mask_token[-1]

		for lang in tqdm(languages):
			if 'en' in lang:
				prompts[lang] = base_prompts[lang]
				continue
			else:
				prompts[lang] = {}

			for label_type in base_prompts['en']:

				prompt, pos_label, neg_label = base_prompts['en'][label_type]
				prompt = preprocess_prompts(prompt,  mask_token)

				try:
					prompt_translation 		= translate(prompt, lang, mask_token)
					pos_label_translation 	= translate(pos_label, lang)
					neg_label_translation	= translate(neg_label, lang)
				except:
					prompt_translation 		= ts.bing(prompt, from_language='en', to_language=lang)
					pos_label_translation 	= ts.bing(pos_label, from_language='en', to_language=lang)
					neg_label_translation	= ts.bing(neg_label, from_language='en', to_language=lang)


				translated_metadata = [prompt_translation, pos_label_translation, neg_label_translation]


				if prompt_translation is not None:
					translated_metadata[0] = prompt_translation.replace(mask_enclosing, mask_token)
					prompts[lang][label_type]  = translated_metadata
				else:
					prompts[lang][label_type] = []

	except RuntimeError as e:
		print("Cannot populate prompts with error: ", e)
		return prompts
	return prompts

def populate_zeroshot(languages: List = [], base_zero: Dict = {}) -> Dict:
	classes_with_sentiments = {}
	try:

		topics, sentiments, hypothesis_template = base_zero['en']['Topics'], base_zero['en']['Sentiments'], base_zero['en']['Hypothesis']

		for lang in tqdm(languages):
			if 'en' in lang:
				classes_with_sentiments[lang] = base_zero[lang]
				continue
			else:
				classes_with_sentiments[lang] = {}

				try:
					topics_translation 				= [translate(topic, lang) for topic in topics]
					sentiments_translation 			= [translate(sentiment, lang) for sentiment in sentiments]
					hypothesis_template_translation	= translate(hypothesis_template, lang)
				except:
					topics_translation 				= [ts.bing(topic, from_language='en', to_language=lang) for topic in topics]
					sentiments_translation 			= [ts.bing(sentiment, from_language='en', to_language=lang) for sentiment in sentiments]
					hypothesis_template_translation	= ts.bing(hypothesis_template, from_language='en', to_language=lang)


				classes_with_sentiments[lang]['Topics'] 	= topics_translation
				classes_with_sentiments[lang]['Sentiments'] = sentiments_translation
				classes_with_sentiments[lang]['Hypothesis'] = hypothesis_template_translation


	except RuntimeError as e:
		print("Cannot populate prompts with error: ", e)
		return classes_with_sentiments
	return classes_with_sentiments




if __name__ == "__main__":

	parser = argparse.ArgumentParser(
	description="Weak Labeler Namespace"
	)

	parser.add_argument(
	'--base_prompts', default = '', type=str,
	help="Path towards the constructed english prompts"
	)

	parser.add_argument(
		'--base_zero', default = '', type=str,
		help="Path towards the constructed english prompts"
		)

	parser.add_argument(
	'--languages',
	help="The path towards the file with languages"
	)

	parser.add_argument(
	'--mask_token', default = '<mask>', type=str,
	help="The mask token used in the model (Note this can be replaced with a simple regexp later)"
	)

	args = parser.parse_args()

	with open(args.languages, 'rb') as file:
		languages = file.readlines()
	languages = [elem.strip().decode("utf-8")  for elem in languages]

	if args.base_prompts:
		with open(args.base_prompts, 'r') as openfile:
			# Reading from json file
			base_prompts = json.load(openfile)


		populated_prompts = populate_prompts(languages, base_prompts, args.mask_token)

		# Serializing json
		json_object = json.dumps(populated_prompts, indent = 4)
		with open("populated_prompts.json", "w") as outfile:
			outfile.write(json_object)

	if args.base_zero:
		with open(args.base_zero, 'r') as openfile:
			# Reading from json file
			base_zero = json.load(openfile)


		populated_prompts = populate_zeroshot(languages, base_zero)

		# Serializing json
		json_object = json.dumps(populated_prompts, indent = 4)
		with open("populated_zero.json", "w") as outfile:
			outfile.write(json_object)
