CUDA_VISIBLE_DEVICES=0 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_bert_ag.json
CUDA_VISIBLE_DEVICES=1 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_bert_acl.json
CUDA_VISIBLE_DEVICES=2 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_bert_res.json
CUDA_VISIBLE_DEVICES=3 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_roberta_ag.json
CUDA_VISIBLE_DEVICES=4 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_roberta_acl.json
CUDA_VISIBLE_DEVICES=5 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_roberta_res.json
CUDA_VISIBLE_DEVICES=6 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_scibert_ag.json
CUDA_VISIBLE_DEVICES=7 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_scibert_acl.json

# TODO
CUDA_VISIBLE_DEVICES=0 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_scibert_res.json

0 ok
77 lost the first and the last

77 ok
42 0 7

42 ok
100 0 7 8

100 ok
237 only 1-5

eval time!
CUDA_VISIBLE_DEVICES=1 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_bert_ag.json
CUDA_VISIBLE_DEVICES=1 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_bert_acl.json
CUDA_VISIBLE_DEVICES=2 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_bert_res.json
CUDA_VISIBLE_DEVICES=3 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_roberta_ag.json
CUDA_VISIBLE_DEVICES=4 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_roberta_acl.json
CUDA_VISIBLE_DEVICES=5 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_roberta_res.json
CUDA_VISIBLE_DEVICES=6 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_scibert_ag.json
CUDA_VISIBLE_DEVICES=7 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_scibert_acl.json
CUDA_VISIBLE_DEVICES=0 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_scibert_acl.json

train adpt!
seed : 0,42,77,100,237
0 ok! 42 ok! 77 ok! 100 ok! 237 ok!
CUDA_VISIBLE_DEVICES=3 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_bert_ag.json
CUDA_VISIBLE_DEVICES=4 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_bert_acl.json
CUDA_VISIBLE_DEVICES=6 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_bert_res.json
CUDA_VISIBLE_DEVICES=7 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_roberta_ag.json
CUDA_VISIBLE_DEVICES=0 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_roberta_acl.json
CUDA_VISIBLE_DEVICES=1 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_scibert_acl.json
CUDA_VISIBLE_DEVICES=2 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_scibert_res.json
CUDA_VISIBLE_DEVICES=3 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_roberta_res.json
CUDA_VISIBLE_DEVICES=0 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_scibert_ag.json


eval adpt! 0 ok! 42 ok! 77 ok! 100 ok! 237
CUDA_VISIBLE_DEVICES=1 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_bert_ag.json
CUDA_VISIBLE_DEVICES=2 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_bert_acl.json
CUDA_VISIBLE_DEVICES=3 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_bert_res.json
CUDA_VISIBLE_DEVICES=4 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_roberta_ag.json
CUDA_VISIBLE_DEVICES=7 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_roberta_acl.json

CUDA_VISIBLE_DEVICES=1 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_roberta_res.json
CUDA_VISIBLE_DEVICES=2 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_scibert_ag.json
CUDA_VISIBLE_DEVICES=3 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_scibert_acl.json
CUDA_VISIBLE_DEVICES=4 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_scibert_res.json
