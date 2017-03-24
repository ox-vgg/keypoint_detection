function demolive_keypoints()
%% Keypoints detection: Human Pose Estimation (Live Demo)
% This is a MatConvNet demo for human pose human estimation.
% Related Work: Belagiannis V., and Zisserman A.,
% Recurrent Human Pose Estimation, FG (2017).
% Contact: Vasileios Belagiannis, vb@robots.ox.ac.uk
% Part of this demo has been written by Andrea Vedaldi
% For further details, visit http://www.robots.ox.ac.uk/~vgg/software/keypoint_detection/

%Set the model name (recpose-iter2 or recposeFT-iter2)
modelName='recposeFT-iter2';

%Setup MatConvNet
run('../matconvnet-b23/matlab/vl_setupnn');

%Add the Web-Camera package
addpath('/Users/belajohn/Documents/MATLAB/SupportPackages/R2016a/toolbox/matlab/webcam/supportpackages');

% Fixed parameters
GPUt=[]; %default test on CPU
opts.imageSize = [248, 248];

GPUon=0;
if numel(GPUt)>0
  GPUon=1;
end

% Load model
load(modelName, 'net');
net = dagnn.DagNN.loadobj(net) ;

% Prediction Layer
pred={'prediction10'};

if GPUon
  gpuDevice(GPUt);
  net.move('gpu');
else
  net.move('cpu');
end

% web-camera
cam = webcam(1);

start = tic ;
[img,time0] = snapshot(cam);
n = 0 ;

figure(1) ; clf ; hold all ;
h = opts.imageSize(1) ;
w = opts.imageSize(2) ;
fig.axis = gca ;
fig.keypoints = [1:16] ;
fig.bg = surface([1 w ; 1 w], [h h ; 1 1], zeros(2), ...
  'facecolor', 'texturemap', 'cdata', img, ...
  'facealpha', 1, ...
  'edgecolor', 'none') ;
fig.colors = jet(numel(fig.keypoints)) ;
for k = 1:numel(fig.keypoints)
  icol = repmat(reshape(fig.colors(k,:),1,1,3),[62 62]) ;
  fig.fg{k} = surface([1 w ; 1 w], [h h ; 1 1], zeros(2) + k, ...
    'facecolor', 'texturemap', 'cdata', icol, ...
    'facealpha', 'texturemap', 'alphadata', 0.5*ones(62,62,1), ...
    'edgecolor', 'none') ;
end
set(fig.axis,'fontsize', 18) ;
axis equal off ;
xlim([1 w]) ;
ylim([1 h]) ;

while true

  % load an image
  elapsed = toc(start) ;
  img = snapshot(cam);

  d = size(img,1)-size(img,2) ;
  dy = floor(max(d,0)/2) ;
  dx = floor(max(-d,0)/2) ;
  img = img(dy+1:end-dy, dx+1:end-dx, :) ; % center crop
  img = imresize(img,opts.imageSize, 'bilinear') ;
  img = single(img)/256 ;
  im_ = img - 0.5 ;
  
  if GPUon
    im_ = gpuArray(im_);
  end
  
  %evaluate the image
  net.mode='test';
  net.eval({'input', im_}) ;
  
  %gather the requested predictions
  output = cell(numel(pred,1));
  for i=1:numel(pred)
    output{i} = net.vars(net.getVarIndex(pred{i})).value ;
  end
  
  % plot
  set(fig.bg, 'cdata', img) ;
  for k = 1:numel(fig.keypoints)
    map = output{1}(:,:,k) ;
    map = map / max(map(:)) ;
    set(fig.fg{k}, 'alphadata', map) ;
  end
  
  elapsed = toc(start) ;
  n = n + 1 ;
  title(sprintf('Keypoint Detection (%.1f Hz)', n/elapsed)) ;
  drawnow ;
end

end