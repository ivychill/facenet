#!/bin/sh
echo "---------run facenet-----------"
#CUDA_VISIBLE_DEVICES=2 python recognition_main.py
CUDA_VISIBLE_DEVICES=2 python recognition_facenet_main.py
