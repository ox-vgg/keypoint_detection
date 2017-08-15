function setup_keypoint_detection()
%SETUP_KEYPOINT_DETECTION Sets up keypoint_detection by adding its folders 
% to the Matlab path

  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/model-train'], [root '/dagnetworks']) ;
  addpath([root '/core']) ;
