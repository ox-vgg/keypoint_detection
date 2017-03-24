function [net, info] = cnn_regressor_dag(varargin)

%Create the imdb and train the model

% Dataset
opts.datas='BBC';

% Network input resolution
opts.patchHei=120;
opts.patchWi=80;

% Camera (always 1 for this setup)
opts.cam=1;

% Augmentation settings
opts.aug=0;
opts.NoAug=0;

% Export directory for model and imdb
opts.expDir = sprintf('/data/vb/Temp/%s-baseline%d',opts.datas,opts.cam) ;
opts.imdbPath = fullfile(opts.expDir, sprintf('imdb%d.mat',opts.cam));

opts.train.batchSize = 256 ;
opts.train.numSubBatches = 1;
opts.train.numEpochs = 100 ;
opts.train.continue = true ;
opts.train.derOutputs= {'objective', 1} ;
opts.train.learningRate = [0.001*ones(1, 17) 0.0005*ones(1, 50) 0.002*ones(1, 500)  0.03*ones(1, 130) 0.01*ones(1, 100)] ;
opts.train.momentum=0.9;
opts.useBnorm = false ;
opts.batchNormalization = 0;
opts.train.prefetch = false ;

%GPU
opts.train.gpus = [];

% Architecture parameters
opts.initNet='/home/bazile/Temp/data/tukey0.mat'; %pre-trained network
opts.outNode=14;%14 bbc, 18,28,42
opts.outPairNode=8;% pairwise terms
opts.outCombiNode=5;
opts.inNode=3;
opts.lossFunc='tukeyloss-heatmap';
opts.lossFunc2='tukeyloss-pairwiseheatmap';
opts.lossFunc3=[];
opts.errMetric = 'mse-combo';
opts.train.thrs=0;
opts.train.refine=false;
opts.HighRes = 0; %high resolution output
opts.ConcFeat=768;  %number of channels at concat
opts.skip_layer = ''; %skip layer
opts.train.hardNeg=0;%hard negative mining

% Dataset (train and validation files)
opts.DataMatTrain=sprintf('/mnt/ramdisk/vb/%s/%s_imdbsT%daug%d.mat',opts.datas,opts.datas,opts.cam,opts.aug);
opts.DataMatVal=sprintf('/mnt/ramdisk/vb/%s/%s_imdbsV%daug%d.mat',opts.datas,opts.datas,opts.cam,opts.aug);
opts.DataMatTrain2=[]; %for combination of different datasets

% IMDB generation function
opts.imdbfn= [];

% Batch parameters
bopts.numThreads = 15;
bopts.transformation = 'f5' ;
bopts.averageImage = single(repmat(128,1,1,opts.inNode));
bopts.imageSize = [120, 80] ;
bopts.border = [10, 10] ;
bopts.heatmap=0;
bopts.trf=[];
bopts.sigma=[];
bopts.HeatMapSize=[];
bopts.flipFlg='bbc';%full, bbc
bopts.inOcclud=1; %include occluded points
bopts.multipInst=1; %include multiple instances in the heatmaps
bopts.HeatMapScheme=1; %how to generate heatmaps
bopts.rotate=0;%rotation augm.
bopts.scale=0;%scale augm.
bopts.color=0;%color augm.
bopts.pairHeatmap=0;
bopts.bodyPairs = [];
bopts.ignoreOcc=0;%requires 
bopts.magnif=8;%amplifier for the body heatmaps
bopts.facX=0.15;%pairwise heatmap width
bopts.facY=0.08;%pairwise heatmap height

% Parse settings
[opts, trainParams] = vl_argparse(opts, varargin); %main settings
[opts.train, boptsParams]= vl_argparse(opts.train, trainParams); %train settings
[bopts, netParams]= vl_argparse(bopts, boptsParams); %batch settings
net=netParams{1}.net; %network
clear trainParams boptsParams netParams;

opts.train.bodyPairs = bopts.bodyPairs;%structured prediction training
opts.train.trf =  bopts.trf;%transformation from the input to the output space

useGpu = numel(opts.train.gpus) > 0 ;
bopts.GPU=useGpu;

%Paths OSX / Ubuntu
opts.train.expDir = opts.expDir ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath)
    imdb = load(opts.imdbPath);
else
    imdb = opts.imdbfn(opts);
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb','-v7.3') ;
end
       
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

fn = getBatchDagNNWrapper(bopts,useGpu) ;
  
info = cnn_train_dag_reg(net, imdb, fn, opts.train) ;

% -------------------------------------------------------------------------
function fn = getBatchDagNNWrapper(opts, useGpu)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchDagNN(imdb,batch,opts,useGpu) ;

% -------------------------------------------------------------------------
function inputs = getBatchDagNN(imdb, batch, opts, useGpu)
% -------------------------------------------------------------------------

[im, lab] = cnn_regressor_get_batch(imdb, batch, opts, ...
                            'prefetch', nargout == 0) ;
if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  inputs = {'input', im, 'label', lab} ;
end