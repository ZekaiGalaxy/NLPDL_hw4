import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import datasets
import numpy as np
from datasets import load_dataset
from dataHelper import get_dataset
import evaluate
import transformers
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
import wandb

'''
	initialize logging, seed, argparse...
'''
# argparse
@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

@dataclass
class ModelArguments:
    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    load_ckpt_path: Optional[str] = field(
        default="", metadata={"help": "load your checkpoint"}
    )

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments)) 
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
training_args.output_dir+=f'/{model_args.model_name}_{data_args.dataset_name}_{training_args.seed}'
training_args.run_name = f'{model_args.model_name}_{data_args.dataset_name}_{training_args.seed}'

# report_to="wandb", [batch_size, epoch, lr, seed]

# logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    filename=f'/home/zhangzekai/nlpdl_hw4/logs/train_{model_args.model_name}_{data_args.dataset_name}_{training_args.seed}.log'
)
logger.info(f"Training/evaluation parameters {training_args}")

# seed
set_seed(training_args.seed)

'''
	load datasets
'''
raw_datasets = get_dataset(data_args.dataset_name)

label_list = raw_datasets["train"].unique("labels")
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)

'''
	load models
'''

if model_args.model_name == 'roberta':
    model_path = '/home/zhangzekai/Model/roberta-base'
elif model_args.model_name == 'bert':
    model_path = '/home/zhangzekai/Model/bert'
elif model_args.model_name == 'scibert':
    model_path = '/home/zhangzekai/Model/scibert'

if model_args.load_ckpt_path!="":
    model_path = model_args.load_ckpt_path

config = AutoConfig.from_pretrained(
    model_path,
    num_labels=num_labels,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    config=config,
)

'''
	process datasets and build up datacollator
'''
# preprocess
def preprocess_function(examples):
    result = tokenizer(examples["text"], padding=True)
    result["labels"] = examples["labels"] 
    # print(result)
    return result

raw_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
)

train_dataset = raw_datasets["train"]
test_dataset = raw_datasets["test"]

# data collator
# data_collator = default_data_collator
data_collator = DataCollatorWithPadding(tokenizer)

'''
    Evaluate
'''
# https://huggingface.co/evaluate-metric
# micro_f1, macro_f1, accuracy
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_preds):
    predictions = eval_preds.predictions
    references = eval_preds.label_ids
    predictions = np.argmax(predictions, axis=-1)
    results = {}
    results['accuracy'] = acc_metric.compute(predictions=predictions, references=references)
    results['micro_f1'] = f1_metric.compute(predictions=predictions, references=references, average="micro")
    results['macro_f1'] = f1_metric.compute(predictions=predictions, references=references, average="macro")
    return results

'''
    Trainer
'''

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

if training_args.do_train:
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()  
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if training_args.do_predict:
    predicted = trainer.predict(test_dataset)
    results = compute_metrics(predicted)
    print(results)










