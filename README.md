# Keypoint Detection
Our model uses a convolutional neural network with a recurrent module to detect body keypoints from a single image. Below, we provide code to train and test the model.

For a web-based demo visit the [project page](http://www.robots.ox.ac.uk/~vgg/software/keypoint_detection/).

### Software Dependencies
You need MATLAB and the [MatConvNet](https://github.com/vlfeat/matconvnet) toolbox to run this demo. This demo has been compiled and tested for Matlab 2016a, CUDNN v5.1 and cuda 8.0 (using a Linux machine).

### Train Model (MPII Human Pose Dataset)
1. Download and prepare the dataset by executing **getMPIIData_v3.m** and then **splitMPIIData_V4.m**.
	- The data preparation takes several minutes to be completed. The output is the train (**MPI_imdbsT1aug0.mat**, ~3GB) and validation (**MPI_imdbsV1aug0.mat**,  ~0.3GB) files.   
2. Execute **trainBodyPose_example.m** (need to set some parameters such as MatConvNet path and GPU index).
3. To test model, run the demo that follows is explained below.

Parameters for training a model with one recurrent iteration:
```
net = initialize3ObjeRecFusion(opts,2,0,'shareFlag',[0,1]);

opts.derOutputs = {'objective1', 1,'objective2', 1, 'objective4', 1,'objective5', 1, 'objective7', 1,'objective8', 1};

```

Parameters for training a model with two recurrent iterations (**default**):
```
net = initialize3ObjeRecFusion(opts,3,0,'shareFlag',[0,1,1]);

opts.derOutputs = {'objective1', 1,'objective2', 1, 'objective4', 1,'objective5', 1, 'objective7', 1,'objective8', 1, 'objective10', 1,'objective11', 1};

```

### Run Demo
 1. Download a [pre-trained model](https://github.com/ox-vgg/keypoint_models/tree/master/models) if you haven't trained one.
 2. Execute **demo_keypoint.m**. 

### Run Live Demo
You need to have the **Web-Camera** package installed and set the path to it.

 1. Download a [pre-trained model](https://gitlab.com/vggdemo/keypoint_matlab_demo/repository/archive.zip?ref=master) if you haven't trained one.
 2. Execute **demolive_keypoints.m**. 

## Citation

    @inproceedings{Belagiannis17,
            title={Deeper depth prediction with fully convolutional residual networks},
            author={Belagiannis, Vasileios and Zisserman, Andrew},
            booktitle={International Conference on Automatic Face and Gesture Recognition},
            year={2017},
            organization={IEEE}
    }


For further details on the model and web-based demo, please visit our [project-page](http://www.robots.ox.ac.uk/~vgg/software/keypoint_detection/).

## License

Copyright (c) 2017, University of Oxford
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
