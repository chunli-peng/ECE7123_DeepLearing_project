# dl7123_2022S_project2
project2's repo for dl7123： KITTI Dataset Semantic Segmentation

target paper's repo: https://github.com/PRBonn/lidar-bonnetal 

model adapted from resnet50 is implemented on our own

model adapted from hrnet is implemented based on: https://github.com/HRNet/HRNet-Semantic-Segmentation

model adapted from resnest50 is implemented based on: https://github.com/zhanghang1989/ResNeSt

model adapted from convnext is implemented based on: https://github.com/facebookresearch/ConvNeXt

SemanticKITTI dataset: http://semantic-kitti.org/dataset.html

The entire KITTI dataset is 80G. We trained on sequences 01, 02, 04, 06, 07, 10 and validated on sequence 09.

There are 6 datapoints from sequence 04 uploaded as examples. To train the model use these examples to verify code works: \
`cd tasks/semantic; 
python train.py -d path/to/dataset -ac config/arch/xxx.yaml -l path/to/log -t fake`\
To train the model use real data:\
`cd tasks/semantic; 
python train.py -d DATAPATH -ac config/arch/xxx.yaml -l path/to/log -t real`

Trained model is stored in: `tasks/semantic/path/to/log`\
Trained result is stored in tensorboard: `tasks/semantic/path/to/log/xxx_tb`

:tada::tada::tada:Summer is Coming:tada::tada::tada:
