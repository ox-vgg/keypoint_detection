# Keypoint Detection

This code detects keypoints (hand, elbow, etc) for human pose estimation. The provided script for training can be updated to detect any type of keypoints (like road junction) in an image. This code has been released as a part of the [seebibyte project](http://seebibyte.org).

## Pre-trained model
For the purpose of demonstration, we have provided pre-trained models. To run the demonstration code:
 * Execute the matlab script [examples/testModel.m](examples/testModel.m)
 * Chose an input image (some sample images are provided in [examples/img/](examples/img/) folder)
 * The pretrained model is an implementation of the Fusion network from [Pfister et al (ICCV 2015)](http://arxiv.org/abs/1506.02897">http://arxiv.org/abs/1506.02897).


## Training
We have a [step-wise guide](https://htmlpreview.github.io/?https://github.com/ox-vgg/keypoint-detection/blob/master/ReadMe.html) to help you train the provided model on your own dataset. Here is a quick summary:
 * Prepare your training (dataset/Train.mat) and validation (dataset/Validation.mat) data set. If you need help preparaing this dataset, please contact [Vasileios Belagiannis](mailto:vb@robots.ox.ac.uk).
 * Update the training parameters in [examples/trainModel.m](examples/trainModel.m) (if required)
 * Run the matlab script [examples/trainModel.m](examples/trainModel.m)

If you have questions, please contact [Vasileios Belagiannis](mailto:vb@robots.ox.ac.uk) and if you are unable to run the provided scripts, please [create an issue](https://github.com/ox-vgg/keypoint-detection/issues/new).


