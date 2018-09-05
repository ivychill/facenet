#!/bin/bash
help_info ()
{
	echo "****************************************************************************"
	echo "*"
	echo "* MODULE:             Linux Script - facenet"
	echo "*"
	echo "* COMPONENT:          This script used to run facenet python process"
	echo "*"
	echo "* REVISION:           $Revision: 1.0 $"
	echo "*"
	echo "* DATED:              $Date: 2018-04-09 15:16:28 +0000 () $"
	echo "*"
	echo "* AUTHOR:             PCT"
	echo "*"
	echo "***************************************************************************"
	echo ""
	echo "* Copyright yanhong.jia@kuang-chi.com. 2020. All rights reserved"
	echo "*"
	echo "***************************************************************************"
}

usage()
{
    echo "##################Usage################## "
    echo "You provided $# parameters,but 3 are required. \
    th first parameter is run cmd,\
    the second parameter is datasets path,\
    the third parameter is project path"
    #echo " facenet pretrain  use: ./facenet.sh pretrain "
    #echo "facenet train on own images: ./facenet.sh train "
    echo "examples are as follows:"

    echo "./facenet_train.sh represent data_dir "
    echo "./facenet_train.sh facenet_eval test_dir "
    echo "./facenet_train.sh facenet_train_mmdinput train_dir_ID train_dir_camera valid_dir"
    echo "./facenet_train.sh facenet_train train_dir_ID train_dir_camera valid_dir"



}

case $1 in

    represent)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES=1
        python util/batch_represent_double.py  \
        --data_dir $2  --output_dir $3 \
        --trained_model_dir $(pwd)/models/double_input_mmd/20180831-134022
        ;;
 
    facenet_eval)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES=2,3
        echo "my_validate_on_lfw ....."
        python evaluation/my_validate_on_lfw_double.py \
        --lfw_dir $2 \
        --model $(pwd)/models/double_input_mmd/20180829-152222 \
        --lfw_pairs $2/pairs.txt \
        ;;
    facenet_train_mmdinput)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES=2
        echo "train_tripletloss ....."
        python src/train_tripletloss_mmd.py \
          --logs_base_dir ./logs \
          --models_base_dir ./models/double_input_mmd \
          --data_dir_ID $2 \
          --data_dir_camera $3 \
          --model_def models.inception_resnet_v1_double \
          --pretrained_model ./models/20170512-110547/model-20170512-110547.ckpt-250000 \
          --optimizer ADAM \
          --learning_rate 0.005 \
          --learning_rate_decay_epochs 10 \
          --learning_rate_decay_factor 0.8 \
          --weight_decay 1e-4 \
          --lfw_dir $4 \
          --lfw_pairs $4/pairs.txt \
          --alpha 0.2 \
          --max_nrof_epochs 5000  \
          --people_per_batch 45 \
          --images_per_person 10 \
          --people_per_batch_mmd 10 \
          --images_per_person_mmd 8 \
          --gpu_memory_fraction 1
        ;;
    facenet_train)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES=2
        echo "train_tripletloss ....."
        python src/train_tripletloss_double_mmd.py \
          --logs_base_dir ./logs \
          --models_base_dir ./models/double_input_mmd \
          --data_dir_ID $2 \
          --data_dir_camera $3 \
          --model_def models.inception_resnet_v1_double \
          --pretrained_model ./models/double_input/20180822-184740/model-20180822-184740.ckpt-160101 \
          --optimizer ADAM \
          --learning_rate 0.01 \
          --learning_rate_decay_epochs 20 \
          --learning_rate_decay_factor 0.8 \
          --weight_decay 1e-4 \
          --lfw_dir $4 \
          --lfw_pairs $4/pairs.txt \
          --alpha 0.2 \
          --max_nrof_epochs 5000  \
          --people_per_batch 45 \
          --images_per_person 10 \
          --gpu_memory_fraction 1
        ;;

    clear)
        find . -name "*.pyc" -type f -print -exec rm -rf {} \;
    ;;
    *)
		help_info
	    usage
		exit 1
    ;;
esac
exit 0