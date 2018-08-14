#!/bin/bash

SECONDS=0

# Setup directories
DATA_DIR='/data/carvalhj/wikitext/shard/'
DATASET_BYTES=534929548  # size of the full dataset (in bytes)
SAVE_DIR='/output/save/'
LOG_DIR='/output/logs/'

# Shards
# Number of shards
TOTAL_SHARDS=98 
INITIAL_TRAIN_SHARD=0
FINAL_TRAIN_SHARD=95
VALID_SHARD=96

let "NUM_TRAIN_SHARDS = $FINAL_TRAIN_SHARD - $INITIAL_TRAIN_SHARD + 1"

# MODEL parameters
RNN_SIZE=2048
BATCH_SIZE=128
SEQ_LENGTH=128
NUM_EPOCHS=3
LR=0.0005
PRINT_EVERY=100

let "TRAINING_BYTES = $NUM_EPOCHS * $DATASET_BYTES * $NUM_TRAIN_SHARDS/$TOTAL_SHARDS"

for i in $(seq $INITIAL_TRAIN_SHARD $FINAL_TRAIN_SHARD); do
    if (( $i == $INITIAL_TRAIN_SHARD )); then
        # First training shard.
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
         --train_bytes=$TRAINING_BYTES \
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
         --train_bytes=$TRAINING_BYTES \
         --print_every=$PRINT_EVERY
    fi

    echo
    echo ">>>>>----------------------Validation shard "$VALID_SHARD
    python3 train.py \
     --option='validate' \
     --data_dir=$DATA_DIR \
     --save_dir=$SAVE_DIR \
     --log_dir=$LOG_DIR \
     --shard=$VALID_SHARD \
     --init_from=$SAVE_DIR$i\
     --rnn_size=$RNN_SIZE \
     --batch_size=$BATCH_SIZE \
     --seq_length=$SEQ_LENGTH \
     --num_epochs 1\
     --lr_init=$LR 

done


echo
echo ">>>>>----------------------Validation shard "$VALID_SHARD
python3 train.py \
 --option='validate' \
 --data_dir=$DATA_DIR \
 --save_dir=$SAVE_DIR \
 --log_dir=$LOG_DIR \
 --shard=$VALID_SHARD \
 --init_from=$SAVE_DIR$FINAL_TRAIN_SHARD \
 --rnn_size=$RNN_SIZE \
 --batch_size=$BATCH_SIZE \
 --seq_length=$SEQ_LENGTH \
 --num_epochs 1\
 --lr_init=$LR 


# Elapsed time
elapsed_t=$SECONDS
echo
echo "TOTAL time: $(($elapsed_t / 3600)):$((($elapsed_t / 60) % 60)):$(($elapsed_t % 60)) - hh:mm:ss"
