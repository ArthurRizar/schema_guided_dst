#coding:utf-8
###################################################
# File Name: start.sh
# Author: Meng Zhao
# mail: @
# Created Time: 2020年01月02日 星期四 13时35分54秒
#=============================================================
source activate tensorflow_new_3.6
export MODEL_DIR=/root/zhaomeng/google-BERT/uncased_L-12_H-768_A-12
#export MODEL_DIR=/root/zhaomeng/google-BERT/chinese_L-12_H-768_A-12
#export MODEL_DIR=/root/zhaomeng/baidu_ERNIE/pad_to_tf/checkpoints
#export MODEL_DIR=/home/zhaomeng/roBerta_model/HIT

export DATA_DIR=/root/zhaomeng/dst_test/dstc8-schema-guided-dialogue

export OUTPUT_DIR=output

#10000,20000,103235, 206470
export CKPT_NUM=206470
export PREDICTION_DIR=$OUTPUT_DIR/checkpoints/pred_res_"$CKPT_NUM"_dev_dstc8_single_domain_dstc8-schema-guided-dialogue

python -m evaluate \
        --dstc8_data_dir $DATA_DIR \
        --prediction_dir $PREDICTION_DIR \
        --eval_set dev \
        --output_metric_file $OUTPUT_DIR/output_metric_file
