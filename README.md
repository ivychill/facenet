本project基于论文[FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832)的[官方（原作）tensorflow实现](https://github.com/davidsandberg/facenet/)

对原作作了一系列优化，包括：
* loss. 原作支持supervised learning，如triplet loss和softmax loss. 本项目在supervised learning的基础上同时支持以domain adaptation为目的的unsupervised learning, 如[mmd](http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf), [dann](https://arxiv.org/abs/1505.07818)和[associative](https://arxiv.org/abs/1708.00938).
* data. 原作只有1个数据供给队列. 本项目有监督和非监督2个队列, 队列支持多数, 支持多种，如混合供给, 轮流供给.
* 

训练、验证等各种脚本都在facenet.sh
## 训练
```bash
./facenet.sh train
```
### 修改训练参数
当你需要修改训练参数时，修改文件facenet.sh的以下这些行
```bash
    train)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES=1
        echo "train_tripletloss ....."
        python src/main_tripletloss.py \
            --logs_base_dir ./logs \
            --models_base_dir ./models/ \
            --data_dir ./data \
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
            --gpu_memory_fraction 1.0
        ;;
```
### 修改loss
当你需要修改训练参数时，修改文件triplet.py的以下这些行
```python
        # multiplier 1.0 may not be the best
        domain_adaptation_loss = 1.0 * losses.mmd_loss(source_end_points['PreLogitsFlatten'], target_end_points['PreLogitsFlatten'], 1.0)
        tf.add_to_collection('losses', domain_adaptation_loss)
        # end: domain adaptation loss
```