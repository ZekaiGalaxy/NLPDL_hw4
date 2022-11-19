# train
CUDA_VISIBLE_DEVICES=0 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_bert_ag.json
CUDA_VISIBLE_DEVICES=1 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_bert_acl.json
CUDA_VISIBLE_DEVICES=2 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_bert_res.json
CUDA_VISIBLE_DEVICES=3 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_roberta_ag.json
CUDA_VISIBLE_DEVICES=4 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_roberta_acl.json
CUDA_VISIBLE_DEVICES=5 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_roberta_res.json
CUDA_VISIBLE_DEVICES=6 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_scibert_ag.json
CUDA_VISIBLE_DEVICES=7 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_scibert_acl.json
CUDA_VISIBLE_DEVICES=8 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/train_scibert_res.json

# eval
CUDA_VISIBLE_DEVICES=1 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_bert_ag.json
CUDA_VISIBLE_DEVICES=1 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_bert_acl.json
CUDA_VISIBLE_DEVICES=2 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_bert_res.json
CUDA_VISIBLE_DEVICES=3 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_roberta_ag.json
CUDA_VISIBLE_DEVICES=4 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_roberta_acl.json
CUDA_VISIBLE_DEVICES=5 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_roberta_res.json
CUDA_VISIBLE_DEVICES=6 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_scibert_ag.json
CUDA_VISIBLE_DEVICES=7 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_scibert_acl.json
CUDA_VISIBLE_DEVICES=8 python /home/zhangzekai/nlpdl_hw4/train.py /home/zhangzekai/nlpdl_hw4/training_args/eval_scibert_acl.json
