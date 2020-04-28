#coding:utf-8
###################################################
# File Name: start.sh
# Author: Meng Zhao
# mail: @
# Created Time: 2020年01月02日 星期四 13时35分54秒
#=============================================================
source activate tensorflow_new_3.6
export MODEL_DIR=/root/zhaomeng/google-BERT/uncased_L-12_H-768_A-12
#export MODEL_DIR=/home/zhaomeng/google_bert_models/uncased_L-12_H-768_A-12
#export MODEL_DIR=/root/zhaomeng/google-BERT/chinese_L-12_H-768_A-12
#export MODEL_DIR=/root/zhaomeng/baidu_ERNIE/pad_to_tf/checkpoints
#export MODEL_DIR=/home/zhaomeng/roBerta_model/HIT

export DATA_DIR=/root/zhaomeng/dst_test/dstc8-schema-guided-dialogue
#export DATA_DIR=../dstc8-schema-guided-dialogue

export OUTPUT_DIR=output_bert

export OUTPUT_EMBEDDING_DIR=embeddings

export OUTPUT_CKPT_DIR=output_bert/checkpoints

#export CKPT_NUM=10000,20000,103235, 206470
export CKPT_NUM=50000

export TASK_NAME=dstc8_single_domain
#export TASK_NAME=dstc8_multi_domain

python -m bert_dst.train_and_predict \
        --bert_ckpt_dir $MODEL_DIR \
        --dstc8_data_dir $DATA_DIR \
        --dialogues_example_dir $OUTPUT_DIR \
        --schema_embedding_dir $OUTPUT_EMBEDDING_DIR \
        --output_dir $OUTPUT_CKPT_DIR \
        --dataset_split dev \
        --run_mode predict \
        --task_name $TASK_NAME \
        --do_lower_case True \
        --eval_ckpt $CKPT_NUM

