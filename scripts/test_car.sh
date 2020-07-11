#!/bin/csh

echo "Starting test"
python3.7 test.py --dataroot ./datasets/car --name DeepSIMCar --label_nc 0 --no_instance --apply_binary_threshold 0 --tps_aug 0
echo "Finish running"
