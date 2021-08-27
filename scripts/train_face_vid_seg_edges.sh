#!/bin/csh
echo "Starting train"
python3.7 ./train.py --dataroot ./datasets/face_video --primitive seg_edges --no_instance --tps_aug 1 --name DeepSIMFaceVideo --test_canny_sigma 0.5
echo "Finish running"
