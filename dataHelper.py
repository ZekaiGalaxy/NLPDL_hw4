def get_dataset(dataset_name, sep_token):
	'''
	dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	'''
	dataset = None

	# your code for preparing the dataset...

	return dataset

import json
import pandas as pd
from datasets import Dataset, DatasetDict
import numpy as np

def load_json(path):
    with open(path, mode='rb') as f:
        return json.load(f)

def load_jsonl(path):
    data = []
    with open(path, mode='rb') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def fetch_list(lst, indices):
	ans = [lst[x] for x in indices]
	return ans

def fs_postprocess(train_data, name):
	# train_text, train_labels
	text = train_data['text']
	labels = train_data['labels']

	if name in ['restaurant', 'laptop']:
		n_class = 3
		total = 32
	elif name == 'agnews':
		n_class = 4
		total = 32
	elif name == 'acl':
		n_class = 6
		total = 48
	
	label = []
	for i in range(n_class):
		label.append([])
	
	for i in range(len(labels)):
		label[labels[i]].append(i)
	
	if name in ['restaurant', 'laptop']:
		choices = []
		choices.extend(list(np.random.choice(label[0],11)))
		choices.extend(list(np.random.choice(label[1],11)))
		choices.extend(list(np.random.choice(label[2],10)))

		text, labels = fetch_list(text, choices), fetch_list(labels, choices)

	elif name == 'agnews':
		choices = []
		for i in range(4):
			choices.extend(list(np.random.choice(label[i],8)))

		text, labels = fetch_list(text, choices), fetch_list(labels, choices)
	
	elif name == 'acl':
		choices = []
		for i in range(6):
			choices.extend(list(np.random.choice(label[i],8)))

		text, labels = fetch_list(text, choices), fetch_list(labels, choices)
	
	return {'text':text, 'labels':labels}


"""
DatasetDict{
	'train': Dataset({'text':[], 'labels':[]}),
	'text': Dataset({'text': [], 'labels':[]})
}
"""

# text, labels
# semeval
def load_semeval(name, sep_token, mode):
	data = list(load_json(f'/home/zhangzekai/nlpdl_hw4/data/{name}/{mode}.json').values())
	text = [dat['term'] + f' {sep_token} ' + dat['sentence'] for dat in data]
	sentiment2label={
		'negative':0,
		'positive':1,
		'neutral':2
	}
	labels = [sentiment2label[dat['polarity']] for dat in data]
	return {'text':text, 'labels':labels}

# acl-arc
def load_acl_arc(sep_token, mode):
	# text / extended_context / cleaned_cite_text ?
	data = load_jsonl(f'/home/zhangzekai/nlpdl_hw4/data/acl_arc/{mode}.jsonl')
	text = [dat['text'] for dat in data]
	intent2label={
		'Background':0,
		'Motivation':1,
		'CompareOrContrast':2,
		'Future':3,
		'Extends':4,
		'Uses':5,
	}
	labels = [intent2label[dat['intent']] for dat in data]
	return {'text':text, 'labels':labels}

# agnews
def load_agnews(sep_token):
	def preprocess(sent):
		sent = sent.replace('\\\\', f' {sep_token} ').replace('\\',' ')
		return sent
	data = pd.read_csv('/home/zhangzekai/nlpdl_hw4/data/agnews/test.csv')
	text = data['Description'].tolist()
	text = [preprocess(t) for t in text]
	labels = data['Class Index'].tolist()
	labels = [l-1 for l in labels]
	return {'text':text, 'labels':labels}

def get_single_dataset(dataset_name, sep_token='[SEP]'):
	name, version = dataset_name.split('_')
	if not name in ['laptop', 'restaurant', 'acl', 'agnews']:
		return None
	if not version in ['sup', 'fs']:
		return None

	datadict = None

	if name in ['laptop', 'restaurant']:
		train_data = load_semeval(name, sep_token, 'train')
		if version == 'fs':
			train_data = fs_postprocess(train_data, name)
		test_data = load_semeval(name, sep_token, 'test')
		datadict = DatasetDict(
			{
				'train': Dataset.from_dict(train_data),
				'test': Dataset.from_dict(test_data),
			}
		)

	elif name == 'acl':
		train_data = load_acl_arc(sep_token, 'train')
		if version == 'fs':
			train_data = fs_postprocess(train_data, name)
		test_data = load_acl_arc(sep_token, 'test')
		datadict = DatasetDict(
			{
				'train': Dataset.from_dict(train_data),
				'test': Dataset.from_dict(test_data),
			}
		)

	elif name == 'agnews':
		all_data = load_agnews(sep_token)
		datadict = Dataset.from_dict(all_data).train_test_split(test_size=0.1, seed=2022)
		if version == 'fs':
			train_data = {'text':datadict['train']['text'], 'labels':datadict['train']['labels']}
			train_data = fs_postprocess(train_data, name)
			datadict = DatasetDict(
			{
				'train': Dataset.from_dict(train_data),
				'test': datadict['test'],
			}
		)

	return datadict

def get_dataset(dataset_name, sep_token='[SEP]'):
	if isinstance(dataset_name, list):
		def get_label_num(name):
			if ('restaurant' in name) or ('laptop' in name):
				return 3
			elif 'agnews' in name:
				return 4
			elif 'acl' in name:
				return 6
			else:
				return 0

		train_text = []
		train_labels = []
		test_text = []
		test_labels = []
		base = 0
		for dat in dataset_name:
			original_dataset = get_single_dataset(dat, sep_token)
			original_train_text = original_dataset['train']['text']
			original_train_labels = original_dataset['train']['labels']
			original_train_labels = [x+base for x in original_train_labels]
			original_test_text = original_dataset['test']['text']
			original_test_labels = original_dataset['test']['labels']
			original_test_labels = [x+base for x in original_test_labels]
			train_text.extend(original_train_text)
			train_labels.extend(original_train_labels)
			test_text.extend(original_test_text)
			test_labels.extend(original_test_labels)
			base += get_label_num(dat)
		
		return DatasetDict(
			{
				'train': Dataset.from_dict({'text':train_text,'labels':train_labels}),
				'test': Dataset.from_dict({'text':test_text,'labels':test_labels}),
			})


		
	elif isinstance(dataset_name, str):
		return get_single_dataset(dataset_name, sep_token)

# dataset_names = ["restaurant_sup", "laptop_sup", "acl_sup", "agnews_sup", "restaurant_fs", "laptop_fs", "acl_fs", "agnews_fs"]

# print(get_dataset('laptop_sup'))
# print(get_dataset('restaurant_sup'))
# print(get_dataset('acl_sup'))
# print(get_dataset('agnews_sup'))
# print(get_dataset('laptop_fs'))
# print(get_dataset('restaurant_fs'))
# print(get_dataset('acl_fs'))
# print(get_dataset('agnews_fs'))
# print(get_dataset(['laptop_sup','restaurant_fs']))
# print(get_dataset(['acl_sup','agnews_fs']))









