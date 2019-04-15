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
        export CUDA_VISABLE_DEVICES=2
        echo "my_validate_on_lfw ....."
        python evaluation/my_validate_on_lfw.py \
        --lfw_dir $2 \
        --model $(pwd)/models/double_input/20180814-164925 \
        --lfw_pairs $2/pairs.txt \
        ;;

    train)
        export PYTHONPATH=$(pwd)/src
        python src/main_tripletloss.py \
            --logs_base_dir ./logs \
            --models_base_dir ./models/ \
            --data_source MULTIPLE \
            --data_dir ./data \
            --model_def models.inception_resnet_v1 \
            --optimizer ADAM \
            --learning_rate 0.1 \
            --learning_rate_decay_epochs 10 \
            --learning_rate_decay_factor 0.8 \
            --unsupervised NONE \
            --lfw_dir /data/yanhong.jia/datasets/face_recognition/datasets_for_train/valid_35 \
            --lfw_pairs /data/yanhong.jia/datasets/face_recognition/datasets_for_train/valid_35/pairs.txt \
            --val_dir /data/nfs/kc/liukang/face_data/valid_150 \
            --val_pairs /data/nfs/kc/liukang/face_data/valid_150/pairs.txt \
            --max_nrof_epochs 5000  \
            --people_per_batch 60 \
            --images_per_person 10 \
            --gpu_memory_fraction 1.0 \
            --gpu 1
        ;;

    train_hvd)
        export PYTHONPATH=$(pwd)/src
        HOROVOD_TIMELINE=./logs/timeline.json \
            mpirun -np 4 \
            -H localhost:4 \
            -bind-to none -map-by slot \
            -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
            -mca pml ob1 -x HOROVOD_MPI_THREADS_DISABLE=1 \
        python src/main_tripletloss.py \
            --logs_base_dir ./logs \
            --models_base_dir ./models/ \
            --data_source MULTIPLE \
            --data_dir ./data \
            --model_def models.inception_resnet_v1 \
            --optimizer ADAM \
            --learning_rate 0.1 \
            --learning_rate_decay_epochs 10 \
            --learning_rate_decay_factor 0.8 \
            --unsupervised NONE \
            --lfw_dir /data/yanhong.jia/datasets/face_recognition/datasets_for_train/valid_35 \
            --lfw_pairs /data/yanhong.jia/datasets/face_recognition/datasets_for_train/valid_35/pairs.txt \
            --val_dir /data/nfs/kc/liukang/face_data/valid_150 \
            --val_pairs /data/nfs/kc/liukang/face_data/valid_150/pairs.txt \
            --max_nrof_epochs 5000  \
            --epoch_size 10000 \
            --people_per_batch 60 \
            --images_per_person 10 \
            --gpu_memory_fraction 1.0 \
            --gpu 0,1,2,3 \
            --cluster True \
            --nrof_warmup_epochs 0
        ;;

    train_inc)
        export PYTHONPATH=$(pwd)/src
        python src/main_tripletloss.py \
            --logs_base_dir ./logs \
            --models_base_dir ./models/ \
            --data_dir /data/nfs/kc/liukang/face_data/80w_camera/80w_all \
            --model_def models.inception_resnet_v1 \
            --pretrained_model ../../../models/facenet/20170512-110547/model-20170512-110547.ckpt-250000 \
            --optimizer ADAM \
            --learning_rate 0.01 \
            --learning_rate_decay_epochs 10 \
            --learning_rate_decay_factor 0.8 \
            --weight_decay 1e-4 \
            --lfw_dir /data/yanhong.jia/datasets/face_recognition/datasets_for_train/valid_35 \
            --lfw_pairs /data/yanhong.jia/datasets/face_recognition/datasets_for_train/valid_35/pairs.txt \
            --val_dir /data/yanhong.jia/datasets/face_recognition/datasets_for_train/valid_24peo_3D+camera \
            --val_pairs /data/yanhong.jia/datasets/face_recognition/datasets_for_train/valid_24peo_3D+camera/pairs.txt \
            --max_nrof_epochs 5000  \
            --people_per_batch 45 \
            --images_per_person 10 \
            --gpu_memory_fraction 1.0 \
            --gpu 1
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
