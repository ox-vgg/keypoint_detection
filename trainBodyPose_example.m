function trainBodyPose_example(varargin)
%% Keypoints detection: Training on MPII Human Pose Dataset
% This is a MatConvNet demo for human pose human estimation.
% Related Work: Belagiannis V., and Zisserman A.,
% Recurrent Human Pose Estimation, FG (2017).
% Contact: Vasileios Belagiannis, vb@robots.ox.ac.uk

%% Read Me
% Before executing the training script, the MPII dataset hat to be
% downloaded using the following scripts:
% 1. Download and prepare MPII Pose Dataset by executing:
% getMPIIData_v3.m and then splitMPIIData_V4.m (may take several minutes)
% To train a model the following has to be defined:
% - MatConvNet directories
% - Number of GPU (opts.gpus)

%% MatConvNet library setup (execute vl_setupnn.m)
run(fullfile(fileparts(mfilename('fullpath')),...
  '..', 'matconvnet-b23','matlab', 'vl_setupnn.m')) ;

%% Pose Example
 
addpath('dagnetworks');
addpath('model-train');

opts.datas='MPI';

%actual input to the network (after augmentation)
opts.patchHei=248;
opts.patchWi=248;

opts.flipFlg='mpi';%model for flipping the joint (flip augmentation)
opts.cam=1;
opts.aug=0;
opts.NoAug=1; %used for calling the correct imdb creation function

opts.batchSize = 20;
opts.numEpochs = 50 ;
opts.learningRate = [0.00001*ones(1, 30) 0.000005*ones(1, 5) 0.000001*ones(1, 50)] ;
opts.batchNormalization = 1;%useful for big networks

%GPU
opts.gpus = [3];

opts.outNode=16;%if heatmaps loss, then number of heatmaps
opts.outPairNode=15;% pairwise terms
opts.inNode=3;
opts.lossFunc='l2loss-heatmap';
opts.lossFunc2='l2loss-pairwiseheatmap';
opts.lossFunc3=[];
opts.ConcFeat=384;  %number of channels at concat
opts.skip_layer = 'layer20'; %skip layer

%export path, imdb store path and location of training / validation data.
opts.expDir = sprintf('../data/v1.00-%s_%s_%d_2Obje3Fus',opts.datas,opts.lossFunc,opts.cam) ;
opts.imdbPath = sprintf('../data/%s-baseline_imdb%d.mat',opts.datas, opts.cam);
opts.DataMatTrain=sprintf('../data/%s_imdbsT%daug%d.mat',opts.datas,opts.cam,opts.aug);
opts.DataMatVal=sprintf('../data/%s_imdbsV%daug%d.mat',opts.datas,opts.cam,opts.aug);

%transformation from input image (248X248) to ouput heatmap(62X62)
trf=[0.25 0 0 ; 0 0.25 0; 0 0 1]; %only scale

%build network - 1 iteration
%net = initialize3ObjeRecFusion(opts,2,0,'shareFlag', [0,1]);

%objectives
%opts.derOutputs = {'objective1', 1,'objective2', 1, ... %feed-forward
%             'objective4', 1,'objective5', 1, ... %iter 0 (not-shared w)
%             'objective7', 1,'objective8', 1};    %iter 1 (shared w)
 


%build network - 2 iterations
net = initialize3ObjeRecFusion(opts,3,0,'shareFlag',[0,1,1]);

%objectives
opts.derOutputs = {'objective1', 1,'objective2', 1, ... %feed-forward
             'objective4', 1,'objective5', 1, ... %iter 0 (not-shared w)
             'objective7', 1,'objective8', 1, ... %iter 1 (shared w)
             'objective10', 1,'objective11', 1};  %iter 2 (shared w)

         
opts.net=net;

opts.numThreads = 15;
opts.transformation = 'f25' ;
opts.averageImage = single(repmat(128,1,1,opts.inNode));
opts.fast = 1;
opts.imageSize = [248, 248] ;
opts.border = [8, 8] ;
opts.bord=[0,0,0,0]; %cropping border

%heatmap setting;
opts.heatmap=1;
opts.trf=trf;
opts.sigma=1.3;
opts.FiltSize=31;
opts.HeatMapSize=[62, 62];
opts.padGTim=[0 0];
opts.rotate=1;%rotation flag
opts.scale=1;%scale augm.
%extra parse settings

%occluded keypoints
opts.inOcclud=1;

%multiple instances
opts.multipInst=0;

%heatmap scheme
opts.HeatMapScheme=1; %how to generate heatmaps

opts.train.momentum=0.95;

opts.negHeat=0;%set to 1 to include negative values for the occlusion
opts.ignoreOcc=0;%set to 1 to include negative values for the occlusion
opts.ignoreRest=0; %quasi single human training

opts.pairHeatmap=1; %generate heatmaps for pairs of body parts
opts.bodyPairs = [1 2 3 4 5 7 8 9 11 12 13 14 14 15 7; 2 3 7 5 6 4 10 10 12 13 8 8 15 16 8]; %full body - MPI

opts.magnif=12;%amplifier for the body heatmaps
opts.facX=0.15;%pairwise heatmap width (def. 0.15)
opts.facY=0.08;%pairwise heatmap height

%imdb generation function
opts.imdbfn =@getImdbNoAug;

opts = vl_argparse(opts, varargin);

%create imdb and train
cnn_regressor_dag(opts);
