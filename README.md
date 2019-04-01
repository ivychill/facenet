# FaceNet
本project基于论文[FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832)的[官方（原作）tensorflow实现](https://github.com/davidsandberg/facenet/)

对原作作了一系列优化, 包括:

* loss. 原作支持supervised learning, 如triplet loss和softmax loss. 本项目在supervised learning的基础上同时支持以domain adaptation为目的的unsupervised learning, 如[mmd](http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf), [dann](https://arxiv.org/abs/1505.07818)和[associative](https://arxiv.org/abs/1708.00938).

* data. 原作只有1个数据供给队列. 本项目有监督和非监督2个队列, 队列支持多数数据来源, 支持多种数据供给方式, 如混合供给, 轮流供给.

* metric. 原作只能简单打印几个性能指标, 如tar, far, roc等. 本项目不仅提供更丰富的指标, 还提供指标图, 而且每个epoch自动保存.

训练、验证等各种脚本都在facenet.sh

## 训练
```
./facenet.sh train
```

### 修改训练参数
当你需要修改训练参数时, 修改文件facenet.sh的以下这些行.
这些参数后面的注释很重要, 务必仔细阅读.
```bash
    train)
        export PYTHONPATH=$(pwd)/src
        export CUDA_VISABLE_DEVICES=1
        echo "train_tripletloss ....."
        python src/main_tripletloss.py
            --logs_base_dir ./logs \	        # 日志目录, 一般不需要修改.
            --models_base_dir ./models/ \	    # 模型保存目录, 一般不需要修改.
            --data_source MULTIPLE \			  # 照片目录是单个还是多个, 下文详解.
            --data_dir ./data \			        # 照片目录, 下文详解.
            --model_def models.inception_resnet_v1 \		# 网络结构, 一般不需要修改.
            --pretrained_model ../../../models/facenet/20170512-110547/model-20170512-110547.ckpt-250000 \	                # 加载的模型文件
            --unsupervise MMD \               # 非监督学习loss, 只可取三者之一: NONE, MMD和DANN.
            --weight_decay 1e-4 \	            # weight_decay regulation的系数
            --optimizer ADAM \	                # 优化器
            --learning_rate 0.01 \	            # 学习率
            --learning_rate_decay_epochs 10 \	# 每隔这么多个epoch, 学习率下降.
            --learning_rate_decay_factor 0.8 \	# 每隔learning_rate_decay_epochs个epoch, 学习率下降的倍数.
            --lfw_dir /data/yanhong.jia/datasets/face_recognition/datasets_for_train/valid_35 \	                            # 与训练异分布的验证集目录
            --lfw_pairs /data/yanhong.jia/datasets/face_recognition/datasets_for_train/valid_35/pairs.txt \                 # 与训练异分布的验证集文件
            --val_dir /data/yanhong.jia/datasets/face_recognition/datasets_for_train/valid_24peo_3D+camera \                # 与训练同分布的验证集文件
            --val_pairs /data/yanhong.jia/datasets/face_recognition/datasets_for_train/valid_24peo_3D+camera/pairs.txt \    # 与训练同分布的验证集文件
            --max_nrof_epochs 5000  \	        # 训练这么多个epoch后终止
            --people_per_batch 45 \		        # 顾名思义, 每batch的人数, 必须为3的倍数. 在内存许可的条件下, 尽可能大.
            --images_per_person 10 \		    # 顾名思义, 每次取数据时取的每人的最大照片数.
            --gpu_memory_fraction 1.0 \	        # 显存的占用比, 一般不需要修改.
        ;;
```

#### 照片目录
关乎以上2个训练参数--data_source和--data_dir, 这2个参数是相关的.
--data_source只可取二者之一: SINGLE和MULTIPLE. 注意, 必须为全大写.
--data_dir为照片存放目录.
如果--data_source是SINGLE, 则--data_dir对应的值下每人一个文件夹, 目录结构如下:
```
data
├── andi
│   ├── andi_0001.png
│   ├── ...
```

如果--data_source是MULTIPLE, 则--data_dir对应的值下有且只有3个子文件夹id和camera和id+camera, 分别存放ID照片和camera照片和混合照片, 这三个子文件夹下的目录结构一样, 都是每人一个文件夹. 整个目录结构如下:

```
data
├── id
│   ├── andi
│   │   ├── andi_0001.png
│   │   ├── ...
├── camera
│   ├── andi
│   │   ├── andi_0001.png
│   │   ├── ...
├── id+camera
│   ├── andi
│   │   ├── andi_0001.png
│   │   ├── ...
```

### 修改loss
当你需要修改loss时, 修改文件main_tripletloss.py的函数main的以下这些行.
```python
# multiplier 1.0 may not be the best
domain_adaptation_loss = 1.0 * losses.mmd_loss(source_end_points['PreLogitsFlatten'], target_end_points['PreLogitsFlatten'], 1.0)
tf.add_to_collection('losses', domain_adaptation_loss)
# end: domain adaptation loss
...
total_loss = tf.add_n([triplet_loss] + [domain_adaptation_loss] + regularization_losses , name='total_loss')
```
例如, 如果你可以决定是否需要domain_adaptation_loss, 以及什么样的domain_adaptation_loss.

### 修改数据供给方式
当你需要修改数据供给方式时, 修改文件triple.py的函数sample_people

## 超参间的关系
经常问到的一个问题是，假设有1千万人的数据，需要多少个epoch才能充分利用这些数据。这是个好问题，涉及各个超参间的关系，而它又与选三元组的算法紧密相关。
假如超参取以下值，
people_per_batch = 45
images_per_person = 10
batch_size = 90
epoch_size = 1000
假设从头开始训练，即没有预训练模型，根据实测数据，开始时45*10=450张照片中，select_triplet能选出1000-2000个合适的三元组。随着训练的进行，符合条件的三元组越来越少。
假设每次select_triplet平均选出500个三元组，而每个batch用30个三元组，则每次select_triplet能用于17个batch。
而每个epoch 1000个batch，则每个epoch需要60个select_triplet，即60*45=2700个人。
假设每次对不同的45人作select_triplet，则覆盖1千万人，大致需要4千个epoch。
