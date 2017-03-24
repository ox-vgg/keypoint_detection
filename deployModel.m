clearvars; close all; clc;

%Clean a trained model from the momentum and training stats.

%MatConvNet library
run(fullfile(fileparts(mfilename('fullpath')),...
    '..','matconvnet-b23','matlab', 'vl_setupnn.m')) ;

model = 'keypoint-v3';
load(sprintf('%s.mat',model));
net = dagnn.DagNN.loadobj(net);
net.move('cpu');
net.rebuild();
save(sprintf('%s.mat',model),'net');
