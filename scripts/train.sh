#!/bin/csh

echo "Starting train"
python3.7 train.py --dataroot ./datasets/face --name DeepSIM --niter 8000 --niter_decay 8000 --label_nc 0 --no_instance --tps_aug 1 --apply_binary_threshold 1
echo "Finish running"



