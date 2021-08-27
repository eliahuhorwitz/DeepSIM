#!/bin/csh
echo "Starting Test"
python3.7 ./test.py --dataroot ./datasets/face --primitive edges --phase "test" --no_instance --name DeepSIMFace
echo "Finish Test"
