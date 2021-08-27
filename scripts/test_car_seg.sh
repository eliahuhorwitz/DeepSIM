#!/bin/csh

echo "Starting Test"
python3.7 ./test.py --dataroot ./datasets/car --primitive seg --phase "test" --no_instance --name DeepSIMCar
echo "Finish Test"



