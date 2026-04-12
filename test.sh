#!/bin/bash
if [ $epoch_time ]; then
    EPOCH_TIME=$epoch_time
else
    EPOCH_TIME=150
fi

if [ $out_dir ]; then
    OUT_DIR=$out_dir
else
    OUT_DIR='./model_out'
fi

if [ $cfg ]; then
    CFG=$cfg
else
    CFG='configs/swin_tiny_patch4_window7_224_lite.yaml'
fi

if [ $data_dir ]; then
    DATA_DIR=$data_dir
else
    DATA_DIR='../data/debris_processed_33'
fi

if [ $learning_rate ]; then
    LEARNING_RATE=$learning_rate
else
    LEARNING_RATE=0.0001
fi

if [ $img_size ]; then
    IMG_SIZE=$img_size
else
    IMG_SIZE=320
fi

if [ $batch_size ]; then
    BATCH_SIZE=$batch_size
else
    BATCH_SIZE=24
fi

echo "start test model"
python test.py --dataset DebrisProcessed --cfg $CFG --root_path $DATA_DIR --output_dir $OUT_DIR --img_size $IMG_SIZE --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE --split test
