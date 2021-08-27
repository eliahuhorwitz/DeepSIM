#!/bin/csh
echo "Starting test"
python3.7 ./test.py --dataroot ./datasets/face_video --primitive seg_edges --phase "test" --no_instance --name DeepSIMFaceVideo --vid_mode 1 --test_canny_sigma 0.5
echo "Finish test"
