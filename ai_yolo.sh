#!/bin/sh
echo "---kill yolo---"
#ps | grep yolo | xargs kill -9
ps | grep yolo
echo "---------yolo-----------"
if [ -n "$1" ]
then
    if [ -n "$2" ]
    then
        echo "run darknet.$1.py"
        CUDA_VISIBLE_DEVICE=$2 python darknet.$1.py
    else
        echo "run darknet.$1.py"
        CUDA_VISIBLE_DEVICE=0 python darknet.$1.py
    fi
fi
