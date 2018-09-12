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

    echo "./assoc.sh represent data_dir "
    echo "./assoc.sh eval test_dir "
    echo "./assoc.sh train train_dir_ID train_dir_camera valid_dir"
}

case $1 in

    represent)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES=0
        python util/batch_represent_double.py  \
        --data_dir /home/asus/fengchen/facenet/data/pairwise/validate \
        --output_dir /home/asus/fengchen/facenet/emb/pairwise/validate \
        --trained_model_dir $(pwd)/models/20180904-172233/
        ;;

    eval)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES=2,3
        echo "my_validate_on_lfw ....."
        python evaluation/my_validate_on_lfw_double.py \
        --lfw_dir $2 \
        --model $(pwd)/models/double_input/20180814-164925 \
        --lfw_pairs $2/pairs.txt \
        ;;


    train_mmd)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES=1
        echo "train_tripletloss ....."
        python src/train_tripletloss_mmd.py \
          --logs_base_dir ./logs \
          --models_base_dir ./models/ \
          --data_dir_ID ./data/pairwise/id/ \
          --data_dir_camera ./data/pairwise/camera/ \
          --data_dir_associative ./data/pairwise/ \
          --model_def models.inception_resnet_v1_double \
          --pretrained_model ./models/20170512-110547/model-20170512-110547.ckpt-250000 \
          --optimizer ADAM \
          --learning_rate 0.01 \
          --learning_rate_decay_epochs 10 \
          --learning_rate_decay_factor 0.8 \
          --weight_decay 1e-4 \
          --lfw_dir data/validate \
          --lfw_pairs data/validate/pairs.txt \
          --max_nrof_epochs 5000  \
          --batch_size 15 \
          --people_per_batch 30 \
          --images_per_person 10 \
          --people_per_batch_assoc 10 \
          --images_per_person_assoc 8 \
          --gpu_memory_fraction 1.0
        ;;


    train_dann)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES=3
        echo "train_tripletloss ....."
        python src/train_tripletloss_dann.py \
          --logs_base_dir ./logs \
          --models_base_dir ./models/ \
          --models_plus_base_dir ./models_plus/ \
          --data_dir_ID ./data/pairwise/id/ \
          --data_dir_camera ./data/pairwise/camera/ \
          --data_dir_associative ./data/pairwise/ \
          --model_def models.inception_resnet_v1_double \
          --pretrained_model ./models/20170512-110547/model-20170512-110547.ckpt-250000 \
          --optimizer ADAM \
          --learning_rate 0.01 \
          --learning_rate_decay_epochs 10 \
          --learning_rate_decay_factor 0.8 \
          --weight_decay 1e-4 \
          --lfw_dir data/validate \
          --lfw_pairs data/validate/pairs.txt \
          --max_nrof_epochs 5000  \
          --batch_size 30 \
          --people_per_batch 30 \
          --images_per_person 10 \
          --people_per_batch_assoc 30 \
          --images_per_person_assoc 10 \
          --gpu_memory_fraction 1.0
        ;;


    train_assoc)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES=1
        echo "train_tripletloss ....."
        python src/train_tripletloss_assoc.py \
          --logs_base_dir ./logs \
          --models_base_dir ./models/ \
          --data_dir_ID ./data/pairwise/id/ \
          --data_dir_camera ./data/pairwise/camera/ \
          --data_dir_associative ./data/pairwise/ \
          --model_def models.inception_resnet_v1_double \
          --pretrained_model ./models/20180831-103659/model-20180831-103659.ckpt-59002 \
          --optimizer ADAM \
          --learning_rate 0.005 \
          --learning_rate_decay_epochs 10 \
          --learning_rate_decay_factor 0.8 \
          --weight_decay 1e-4 \
          --lfw_dir data/validate \
          --lfw_pairs data/validate/pairs.txt \
          --max_nrof_epochs 5000  \
          --batch_size 30 \
          --people_per_batch 45 \
          --images_per_person 10 \
          --people_per_batch_assoc 10 \
          --images_per_person_assoc 10 \
          --gpu_memory_fraction 1.0
        ;;


    train_mmd_assoc)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES=3
        echo "train_tripletloss ....."
        python src/train_tripletloss_mmd_assoc.py \
          --logs_base_dir ./logs \
          --models_base_dir ./models/ \
          --data_dir_ID ./data/pairwise/id/ \
          --data_dir_camera ./data/pairwise/camera/ \
          --data_dir_associative ./data/pairwise/ \
          --model_def models.inception_resnet_v1_double \
          --pretrained_model ./models/20180831-103659/model-20180831-103659.ckpt-59002 \
          --optimizer ADAM \
          --learning_rate 0.005 \
          --learning_rate_decay_epochs 10 \
          --learning_rate_decay_factor 0.8 \
          --weight_decay 1e-4 \
          --lfw_dir data/validate \
          --lfw_pairs data/validate/pairs.txt \
          --max_nrof_epochs 5000  \
          --batch_size 30 \
          --people_per_batch 45 \
          --images_per_person 10 \
          --people_per_batch_mmd 10 \
          --images_per_person_mmd 10 \
          --people_per_batch_assoc 10 \
          --images_per_person_assoc 10 \
          --gpu_memory_fraction 1.0
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
