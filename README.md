# NLPDL_hw4
## Task2 :  Training Script

In this task, you have to write a `train.py` to train a transformer using the dataset prepared from the last task. Here is the reference material which you can follow.

[transformers/run_glue.py at main · huggingface/transformers](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py)

1. Search the relevant document for `HfArgumentParser`, understand how to use it, and use it in your `train.py`. (think: What’s the advantage over `argparse`?)
2. Search the relevant document for `logging`, understand how to use it, and use it in your `train.py`. (think: What’s the advantage over `print`?)
3. Use `set_seed` in your `train.py`. (think: Why should we set the random seed?)
4. Use `get_dataset` defined in `dataHelper.py`.
5. Search the relevant document for `AutoConfig`, `AutoTokenizer`, `AutoModelForSequenceClassification` understand how to use them, and use them in your `train.py`. 
6. use `datasets.map` and `tokenzier` to process the dataset, process the text string into tokenized ids. 
7. search the relevant document for `evaluate`(Huggingface library), understand how to use it, and use it in your `train.py`(you need to compute `micro_f1`, `macro_f1`, `accuracy`).
8. search the relevant docs for `DataCollatorWithPadding`, understand how to use it, and what does it do, and use it in your `train.py`. (think: Are there any other data-collator you can use?)
9. Understand [Trainer](https://huggingface.co/docs/transformers/v4.22.1/en/main_classes/trainer#trainer) , and you can look into the source code by clicking <source>. Use `Trainer` in your `train.py`.
10. Clear up all the redundant code in your `train.py`. If you directly copy code from `huggingface/trainsformers/run_glue.py` above, there will be loads of redundancy. You must make the code really clean.
11. Write annotations in `train.py`. For example,
    
    ```python
    '''
    	initialize logging, seed, argparse...
    '''
    # your code for init relevant componets...
    
    '''
    	load datasets
    '''
    # your code for loading dataset...
    
    '''
    	load models
    '''
    # your code for loading model...
    
    '''
    	process datasets and build up datacollator
    '''
    # your code for processing datasets
    
    trainer = Trainer(...) # build up trainer
    
    # training!
    # your codes for training...
    
    # ...
    ```
    
12.  Use `[wandb](http://wandb.ai)` to track your experiments. It’s a very fantastic tool!!! And it can easily interact with Huggingface, see [https://docs.wandb.ai/guides/integrations/huggingface](https://docs.wandb.ai/guides/integrations/huggingface).
13. Run your `train.py`! The setting are as follows.
    1. Use `roberta-base` , `bert-base-uncased` and `allenai/scibert_scivocab_uncased` as your pre-trained models.
    2. Use `restaurant_sup`, `acl_sup` and `agnews_sup` as your datasets.
    3. To make the results reliable, you need to run the same experiments several times(~5 runs) and report the standard deviation.
    
    Adjust the batch size and epoch number to make your results converge stably. Write a small report to show the learning curves (metrics, loss during training, you can copy from `wandb`), results and configurations. Compare and analyze the results for different models and different datasets in your report.
