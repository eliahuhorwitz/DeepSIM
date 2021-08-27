#!/bin/csh
echo "Starting train"
python3.7 ./train.py --dataroot ./datasets/face_video --primitive seg_edges --no_instance --tps_aug 1 --name DeepSIMFaceVideoAugmentations --cutmix_aug 1 --affine_aug "shearx_sheary_rotation" --canny_aug 1
echo "Finish running"
