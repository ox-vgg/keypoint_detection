function demo_keypoint(varargin)
%DEMO_KEYPOINT Keypoints detection in Human Pose Estimation
%   This is a MatConvNet demo for human pose human estimation.
%   Related Work: Belagiannis V., and Zisserman A.,
%   Recurrent Human Pose Estimation, FG2017.
%   Contact: Vasileios Belagiannis, vb@robots.ox.ac.uk
%   Part of this demo has been written by Abhishek Dutta (adutta@robots.ox.ac.uk)
%   For further details, visit http://www.robots.ox.ac.uk/~vgg/software/keypoint_detection/

  opts.gpus = 1 ;
  opts.img_fn = 'sample_img.jpg' ;
  opts.model = 'keypoint-v4' ;
  opts = vl_argparse(opts, varargin)  ;

  % Initialize keypoint detector
  keypoint_detector = KeyPointDetector(opts.model, opts.gpus) ;

  % Detect keypoints
  fprintf(1, '\nDetecting keypoints in image : %s', opts.img_fn) ;
  [kpx, kpy, kpname] = get_all_keypoints(keypoint_detector, opts.img_fn) ;

  % Display the keypoints
  img = imread(opts.img_fn) ;
  close all  ; figure('Name', 'Detected Keypoints') ;
  imshow(img) ; hold on ;
  plot(kpx, kpy, 'r.', 'MarkerSize', round(size(img,2)/10)) ; hold on ;

  voffset = -10 ;
  for i=1:length(kpname)
    text(double(kpx(i)), double(kpy(i) + voffset), ...
      kpname{i}, 'color', 'yellow', 'FontSize', 8, ...
      'backgroundcolor', 'black') ; 
    hold on ;
    voffset = voffset * -1 ; % to prevent cluttering of annotations
  end
  hold off ;

  fprintf(1, '\nShowing detected keypoints:') ;
  for i=1:length(kpname)
    fprintf(1, '\n%s\tat\t(%d,%d)', kpname{i}, kpx(i), kpy(i)) ;
  end
  fprintf(1, '\n') ;
