#coding:utf-8
###################################################
# File Name: start.sh
# Author: Meng Zhao
# mail: @
# Created Time: 2020年01月02日 星期四 13时35分54秒
#=============================================================
export MODEL_DIR=/root/zhaomeng/google-BERT/uncased_L-12_H-768_A-12
#export MODEL_DIR=/root/zhaomeng/google-BERT/chinese_L-12_H-768_A-12
#export MODEL_DIR=/root/zhaomeng/baidu_ERNIE/pad_to_tf/checkpoints
#export MODEL_DIR=/home/zhaomeng/roBerta_model/HIT

export DATA_DIR=/root/zhaomeng/dst_test/dstc8-schema-guided-dialogue

export OUTPUT_DIR=output

export OUTPUT_EMBEDDING_DIR=embeddings

export OUTPUT_CKPT_DIR=output/checkpoints

python -m baseline.train_and_predict \
        --bert_ckpt_dir $MODEL_DIR \
        --dstc8_data_dir $DATA_DIR \
        --dialogues_example_dir $OUTPUT_DIR \
        --schema_embedding_dir $OUTPUT_EMBEDDING_DIR \
        --output_dir $OUTPUT_CKPT_DIR \
        --dataset_split dev \
        --run_mode predict \
        --task_name dstc8_single_domain \
        --do_lower_case True \
        --eval_ckpt 10000,20000,103235

