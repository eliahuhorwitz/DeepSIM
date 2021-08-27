#!/bin/csh

echo "Starting train"
python3.7 ./train.py --dataroot ./datasets/car --primitive seg --no_instance --tps_aug 1 --name DeepSIMCar
echo "Finish running"



