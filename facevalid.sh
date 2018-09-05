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
    echo "./facevalid.sh align_face /home/yanhong.jia/datasets/facenet_image/valid_name  /home/yanhong.jia/datasets/facenet_image/valid_name_align_160 "
    echo "./facevalid.sh prune /home/yanhong.jia/datasets/facenet_image/valid_name_align_160  3"
    echo "./facevalid.sh represent /home/yanhong.jia/datasets/facenet_image/valid_name_align_160 "
    echo "./facevalid.sh datapair /home/yanhong.jia/datasets/facenet_image/valid_name_align_160 "
    echo "./facevalid.sh facenet_mytrain /home/yiqi.liu-2/yanhong.jia/datasets/facenet_image  /home/yanhong.jia/datasets/lfw-align"
    echo "./facevalid.sh facenet_eval /home/yanhong.jia/datasets/facenet_image/valid_name_align_160 "
    echo "./facevalid.sh facenet_train /home/yiqi.liu-2/yanhong.jia/datasets/facenet_image  /home/yanhong.jia/datasets/lfw-align"
    echo "./facevalid.sh faceKNN_train /home/yiqi.liu-4/yanhong.jia/datasets/face_image/SELFDATA  "
    echo "./facevalid.sh faceknn_test  /home/yiqi.liu-2/yanhong.jia/datasets/facenet_image"
    echo "./facevalid.sh face_test  /home/yiqi.liu-2/yanhong.jia/datasets/facenet_image  "


}

case $1 in
    align_face|s)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES= 0
        echo "align hr dataset 128 -->160"
        for N in {1..3}; do \
        python src/align/align_dataset_mtcnn.py  \
        $2 \
        $3\
        --image_size 160 \
        --margin 12 \
        --random_order \
        --gpu_memory_fraction 0.1 \
        & done
    ;;
    prune)
        python util/prune-dataset.py  $2 \
        --numImagesThreshold $3 \

        ;;
    represent)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES=1
        python util/batch_represent.py  \
        --data_dir $2  --output_dir $3 \
        --trained_model_dir $(pwd)/models/20180519-210715
        ;;
    datapair)
        python evaluation/datapair.py --input_dir $2 \
        --output_dir $2
        ;;
    facenet_eval)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES=2,3
        echo "my_validate_on_lfw ....."
        python evaluation/my_validate_on_lfw.py \
        --lfw_dir $2 \
        --model $(pwd)/models/20180408-102900 \
        --lfw_pairs $2/pairs.txt \
        ;;
    facenet_train)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES=2
        echo "train_tripletloss ....."
        python src/train_tripletloss_double.py \
          --logs_base_dir ./logs \
          --models_base_dir ./models/double \
          --data_dir_ID $2 \
          --data_dir_camera $3 \
          --model_def models.inception_resnet_v1_double \
          --optimizer ADAM \
          --learning_rate 0.001 \
          --weight_decay 1e-4 \
          --lfw_dir $4 \
          --lfw_pairs $4/pairs.txt \
          --max_nrof_epochs 5000  \
          --people_per_batch 45 \
          --images_per_person 10 \
          --gpu_memory_fraction 0.8
        ;;
    facenet_mytrain)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES=2
        echo "train_tripletloss ....."
        python src/my_train_tripletloss.py \
          --logs_base_dir ./logs \
          --models_base_dir ./models \
          --data_dir $2 \
          --pretrained_model ./models/20170512-110547/model-20170512-110547.ckpt-250000 \
          --model_def models.inception_resnet_v1 \
          --optimizer ADAM \
          --learning_rate 0.01 \
          --learning_rate_decay_factor 0.8 \
          --learning_rate_decay_epochs 10 \
          --weight_decay 1e-4 \
          --lfw_dir $3 \
          --lfw_pairs $3/pairs.txt \
          --max_nrof_epochs 500  \
          --people_per_batch 30 \
          --images_per_person 10 \
          --embedding_size 128 \
          --gpu_memory_fraction 0.8
        ;;
    faceKNN_train)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES= 3

        python src/classifier.py TRAIN \
            --data_dir \
            $2 \
            --model \
            $(pwd)/models/20180518-142558 \
            --classifier  KNN \
            --classifier_filename \
            $(pwd)/models/KNN.pkl \
            --batch_size 100
        ;;
    faceknn_test)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES=3

        echo "test facenet ...."
         python evaluation/myclassifier.py CLASSIFY \
        --data_dir  $2 \
        --model $(pwd)/models/20180519-210715 \
        --classifier_filename $(pwd)/models/KNN.pkl  --batch_size 100
        ;;

    face_test)
        export PYTHONPATH=$(pwd)/facenet
        export CUDA_VISABLE_DEVICES=1
        echo "align valid_name_lr_32 dataset 32 -->160"
        python src/align/align_dataset_mtcnn.py \
        $2/valid_name_lr_32  \
        $2/valid_name_lr_mtcnnpy_160 \
        --image_size 160 --margin 32 \
        --random_order --gpu_memory_fraction 0.8
        echo "lr to fsr ..................................."

        python FSRGAN/test_FSRGAN.py  CLASSIFY \
           --lr_data_dir $2/valid_name_lr_32 \
           --lr_bicu_dir $2/valid_name_lr_bicubic_128 \
           --sr_data_dir $2/valid_name_fsr_128
        echo "lr to sr ..................................."
        python  SRGAN/test_SRGAN.py  CLASSIFY \
            --lr_data_dir $2/valid_name_lr_32 \
            --lr_bicu_dir $2/valid_name_lr_bicubic \
            --sr_data_dir $2/valid_name_sr_128



        echo "align sr image dataset 128 -->160"
        python src/align/align_dataset_mtcnn.py \
        $2/valid_name_sr_128  \
        $2/valid_name_sr_mtcnnpy_160 \
        --image_size 160 --margin 32 \
        --random_order --gpu_memory_fraction 0.8

        echo "align fsr image dataset 128 -->160"
        python src/align/align_dataset_mtcnn.py \
        $2/valid_name_fsr_128  \
        $2/valid_name_fsr_mtcnnpy_160 \
        --image_size 160 --margin 32 \
        --random_order --gpu_memory_fraction 0.8

        echo "val bucibuc face......"
        python src/myclassifier.py CLASSIFY \
        --data_dir  $2/valid_name_lr_mtcnnpy_160 \
        --model $(pwd)/models/20180419-140738 \
        --classifier_filename $(pwd)/models/classifier.pkl  --batch_size 100

        echo "val SR face......"
        python src/myclassifier.py CLASSIFY \
        --data_dir $2/valid_name_sr_mtcnnpy_160 \
        --model $(pwd)/models/20180419-140738 \
        --classifier_filename $(pwd)/models/classifier.pkl --batch_size 100

        echo "val FSR face......"
        python src/myclassifier.py CLASSIFY \
        --data_dir $2/valid_name_fsr_mtcnnpy_160 \
        --model $(pwd)/models/20180419-140738 \
        --classifier_filename $(pwd)/models/classifier.pkl --batch_size 100
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