# FaceNet
本project基于论文[FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832)的[官方（原作）tensorflow实现](https://github.com/davidsandberg/facenet/)

对原作作了一系列优化, 包括:

* loss. 原作支持supervised learning, 如triplet loss和softmax loss. 本项目在supervised learning的基础上同时支持以domain adaptation为目的的unsupervised learning, 如[mmd](http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf), [dann](https://arxiv.org/abs/1505.07818)和[associative](https://arxiv.org/abs/1708.00938).

* data. 原作只有1个数据供给队列. 本项目有监督和非监督2个队列, 队列支持多数数据来源, 支持多种数据供给方式, 如混合供给, 轮流供给.

* metric. 原作只能简单打印几个性能指标, 如tar, far, roc等. 本项目不仅提供更丰富的指标, 还提供指标图, 而且每个epoch自动保存.

* batch_size. 众所周知, 由于显存的限制, batch_size是受限的. 本项目提供通过巧妙的方法规避了显存限制, 达到了加大batch_size的等效效果. 

* Clustering. 本项目利用集成了horovod的多机多卡方案. 

训练、验证等各种脚本都在facenet.sh

## 训练
如果是普通的单卡训练, 运行:
```
./facenet.sh train
```
如果是并行的多机多卡训练, 运行:
```
./facenet.sh train_hvd
```

### 修改训练参数
当你需要修改训练参数时, 修改文件facenet.sh的以下这些行.
这些参数后面的注释很重要, 务必仔细阅读.
```bash
    train_hvd)
        export PYTHONPATH=$(pwd)/src
        HOROVOD_TIMELINE=./logs/timeline.json \   # 以下几行仅用于并行
            mpirun -np 8 \              # 总GPU数目
            -H localhost:4 192.168.20.68:4 \            # 指定的主机及其相应的GPU数目.
            -bind-to none -map-by slot \
            -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
            -mca pml ob1 -x HOROVOD_MPI_THREADS_DISABLE=1 \
        python src/main_tripletloss.py
            --logs_base_dir ./logs \	        # 日志目录, 一般不需要修改.
            --models_base_dir ./models/ \	    # 模型保存目录, 一般不需要修改.
            --data_source MULTIPLE \			  # 照片目录是单个还是多个, 下文详解.
            --data_dir ./data \			        # 照片目录, 下文详解.
            --model_def models.inception_resnet_v1 \		# 网络结构, 一般不需要修改.
            --pretrained_model ../../../models/facenet/20170512-110547/model-20170512-110547.ckpt-250000 \	                # 加载的模型文件
            --unsupervised MMD \               # 非监督学习loss, 只可取三者之一: NONE, MMD和DANN.
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
            --people_per_batch 60 \		        # 顾名思义, 每batch的人数, 必须为3的倍数. 在内存许可的条件下, 尽可能大.
            --images_per_person 10 \		    # 顾名思义, 每次取数据时取的每人的最大照片数.
            --gpu_memory_fraction 1.0 \	        # 显存的占用比, 一般不需要修改.
            --gpu 0,1,2,3                   # 绑定的GPU序号, 如果单卡, 则指定1个；如果多卡, 则指定用逗号分隔的多个.
            --cluster True                  # 并行训练, 则为True, 否则为False.
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
经常问到的一个问题是, 假设有1千万人的数据, 需要多少个epoch才能充分利用这些数据. 这是个好问题, 涉及各个超参间的关系, 而它又与选三元组的算法紧密相关. 

假如超参取以下值:
* people_per_batch = 45
* images_per_person = 10
* batch_size = 90
* epoch_size = 10000

假设从头开始训练, 即没有预训练模型. 

根据实测数据, 开始时45(people_per_batch)*10(images_per_person)=450张照片中, select_triplet能选出1000-2000个合适的三元组. 随着训练的进行, 符合条件的三元组越来越少. 

假设每次select_triplet平均选出500个三元组, 而每个batch用30个(batch_size/3)三元组, 则每次select_triplet能用于17个batch. 

假设每个epoch 10000个batch(epoch_size), 则每个epoch需要600个select_triplet, 即600*45=27000个人. 则覆盖1千万人, 大致需要400个epoch. 

根据实测数据, 每epoch耗时2小时, 即一天可训练12个epoch, 则覆盖1千万人, 大致需要35天. 

以上是单卡的情况. 如果是双机总共8卡, 假设可以有10倍的加速效果, 3天即可覆盖1千万人. 
