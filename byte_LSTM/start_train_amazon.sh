#!/bin/bash

SECONDS=0

# Setup directories
DATA_DIR='/data/carvalhj/amazon_reviews/'
SAVE_DIR='/output/save/'
LOG_DIR='/output/logs/'

# Training shards
TOTAL_SHARDS=995
TRAIN_SHARD=993
VALID_SHARD=994

# MODEL parameters
RNN_SIZE=2048
BATCH_SIZE=128
SEQ_LENGTH=128
NUM_EPOCHS=1
LR=0.0005
PRINT_EVERY=200
let "TRAINING_BYTES = 38000000000 * $TRAIN_SHARD/$TOTAL_SHARDS"

for i in $(seq 0 $TRAIN_SHARD); do
    if (( $i == 0 )); then
        echo
        echo "---------------------------Training shard "$i
        python3 train.py \
         --option='train' \
         --data_dir=$DATA_DIR \
         --save_dir=$SAVE_DIR \
         --log_dir=$LOG_DIR \
         --shard=${i} \
         --rnn_size=$RNN_SIZE \
         --batch_size=$BATCH_SIZE \
         --seq_length=$SEQ_LENGTH \
         --num_epochs=$NUM_EPOCHS \
         --lr_init=$LR \
         --lr_decay \
         --total_bytes=$TRAINING_BYTES \
         --print_every=$PRINT_EVERY
    else
        let a=$i-1
        echo
        echo "---------------------------Training shard "$i
        python3 train.py \
         --option='train' \
         --data_dir=$DATA_DIR \
         --save_dir=$SAVE_DIR \
         --log_dir=$LOG_DIR \
         --shard=${i} \
         --init_from=$SAVE_DIR${a} \
         --rnn_size=$RNN_SIZE \
         --batch_size=$BATCH_SIZE \
         --seq_length=$SEQ_LENGTH \
         --num_epochs=$NUM_EPOCHS \
         --lr_init=$LR \
         --lr_decay \
         --total_bytes=$TRAINING_BYTES \
         --print_every=$PRINT_EVERY
    fi

: <<'comment'

    echo
    echo ">>>>>----------------------Validation shard "$VALID_SHARD
    python3 train.py \
     --option='validate' \
     --data_dir=$DATA_DIR \
     --save_dir=$SAVE_DIR \
     --log_dir=$LOG_DIR \
     --shard=$VALID_SHARD \
     --init_from=$SAVE_DIR${i} \
     --rnn_size=$RNN_SIZE \
     --batch_size=$BATCH_SIZE \
     --seq_length=$SEQ_LENGTH \
     --num_epochs=$NUM_EPOCHS \
     --lr_init=$LR 
comment

done

# Elapsed time
elapsed_t=$SECONDS
echo "TOTAL time: $(($elapsed_t / 3600)):$((($elapsed_t / 60) % 60)):$(($elapsed_t % 60)) - hh:mm:ss"
