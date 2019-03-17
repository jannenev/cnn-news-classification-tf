#!/bin/bash

./train3.py \
-filter_sizes 3,5,7 \
-num_filters 128 \
-num_epochs 30 \
-dropout_keep_prob 0.5 \
-batch_size 64 \
-num_epochs
