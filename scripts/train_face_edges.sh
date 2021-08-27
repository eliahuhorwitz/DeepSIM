#!/bin/csh
echo "Starting train"
python3.7 ./train.py --dataroot ./datasets/face --primitive edges --no_instance --tps_aug 1 --name DeepSIMFace
echo "Finish running"
