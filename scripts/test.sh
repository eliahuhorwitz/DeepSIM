#!/bin/csh

echo "Starting test"
python3.7 test.py --dataroot ./datasets/face --name DeepSIM --label_nc 0 --no_instance --resize_or_crop none --apply_binary_threshold 1 --tps_aug 0 --loadSize 640 --fineSize 640
echo "Finish running"
