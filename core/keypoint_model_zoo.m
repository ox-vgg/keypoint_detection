function net = keypoint_model_zoo(modelName)
%KEYPOINT_MODEL_ZOO - load keypoint model by name
%  KEYPOINT_MODEL_ZOO(MODELNAME) - loads a keypoint detector by its given name. 
%  If it cannot be found on disk, it will be downloaded
%
% Copyright (C) 2017 Samuel Albanie and Vasileios Belagiannis
% All rights reserved.

  modelNames = {
    'keypoint-v1', ...
    'keypoint-v2', ...
    'keypoint-v3', ...
    'keypoint-v4', ...
  } ;

  msg = sprintf('%s: unrecognised model', modelName) ;
  assert(ismember(modelName, modelNames), msg) ;
  modelDir = fullfile(vl_rootnn, 'data/models-import') ;
  modelPath = fullfile(modelDir, sprintf('%s.mat', modelName)) ;
  if ~exist(modelPath, 'file'), fetchModel(modelName, modelPath) ; end
  tmp = load(modelPath) ; net = dagnn.DagNN.loadobj(tmp.net) ;

% ---------------------------------------
function fetchModel(modelName, modelPath)
% ---------------------------------------

  waiting = true ;
  msg = ['%s was not found at %s\nWould you like to ', ...
          ' download it from THE INTERNET (y/n)?\n'] ;
  prompt = sprintf(msg, modelName, modelPath) ;

  while waiting
    str = input(prompt,'s') ;
    switch str
      case 'y'
        if ~exist(fileparts(modelPath), 'dir'), mkdir(fileparts(modelPath)) ; end
        fprintf(sprintf('Downloading %s ... \n', modelName)) ;
        baseUrl = 'https://github.com/ox-vgg/keypoint_models/raw/master/models' ;
        url = sprintf('%s/%s.mat', baseUrl, modelName) ;
        urlwrite(url, modelPath) ;
        return ;
      case 'n', throw(exception) ;
      otherwise, fprintf('input %s not recognised, please use `y/n`\n', str) ;
    end
  end
