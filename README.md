# DenseFusionROSNode

## Requirements
   ```
   $ conda env create -f environment.yml --name DFROSNode
   ```

#### Packages
* Ubuntu 18.04
* Cuda 10.0
* Python 2.7: 'conda create --name DFROSNode python=2.7'
* Pytorch 1.4: 'conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch'
* Tensorflow 1.14:  'conda install -c conda-forge tensorflow=1.14'
* Keras 2.1.6: 'conda install keras==2.1.6'

## Launch File
   ```
   $ roslaunch densefusion_ros densefusion_ros.launch
   ```

![Alt text](overview.png?raw=true "Title")
