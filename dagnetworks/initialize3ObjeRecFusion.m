function net = initialize3ObjeRecFusion(opts,Niter,resConn, varargin)
% Related Work: Belagiannis V., and Zisserman A.,
% Recurrent Human Pose Estimation, FG (2017).
% Contact: Vasileios Belagiannis, vb@robots.ox.ac.uk
% The default network is defined to perform 2 iterations
% (i.e. 2 recurrent layers: 1 without shared weights and another with shared)
% Niter: number of iterations (1 - no shared w, 2 - shared w and etc..)
% resConn: Residual connection(not maintained anymore, it should stay to 0)
% To build a network with 1 non-shared and 1 recurrent iteration:
% net = initialize3ObjeRecFusion(opts,2,0,[0,1]);
% To build a network with 1 non-shared and 2 recurrent iterations:
% net = initialize3ObjeRecFusion(opts,3,0,[0,1,1]);

scal = 1 ;
init_bias = 0.0;
net.layers = {} ;
opts.cudnnWorkspaceLimit = 1024*1024*1024*5;
lopts.shareFlag=[0,1];%default: 2 iterations (1. not shared w, 2. shared w)
lopts.ref_idx=[];%reference layer to get the share weights
lopts = vl_argparse(lopts, varargin);
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
id=0;

%define the reference recurrent layer for the shared weights
lopts.ref_idx=0;
i=0;
while i<numel(lopts.shareFlag) &&  lopts.ref_idx==0
    i=i+1;
    if lopts.shareFlag(i)~=0
        lopts.ref_idx=i-1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%SEQUENTIAL PART - START%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Conv 1 - Common
net.layers{end+1} = struct('type', 'conv', ...
    'filters', 0.01/scal * randn(3,3, opts.inNode, 64, 'single'), ...%3X3, 32
    'biases', init_bias*ones(1, 64, 'single'), ...
    'stride', 1, ... %1
    'pad', 1, ...
    'learningRate', [1 2], ...
    'weightDecay', [1 0], ...
    'opts', {convOpts});

if opts.batchNormalization
    id = id + 1;
    out = size(net.layers{end}.filters,4);
    net.layers{end+1} = struct('type', 'bnorm','name', sprintf('bn%d',id),'weights', {{ones(out, 1, 'single'),...
        zeros(out, 1, 'single'), zeros(out, 2, 'single')}}, 'learningRate', [2 1 0.1], ...
        'weightDecay', [0 0]) ;
end

% ReLu 1
net.layers{end+1} = struct('type', 'relu') ;

% Conv 2 - Common
net.layers{end+1} = struct('type', 'conv', ...
    'filters', 0.01/scal * randn(3,3,64, 64, 'single'), ...%3X3, 32
    'biases', init_bias*ones(1, 64, 'single'), ...
    'stride', 1, ... %1
    'pad', 1, ...
    'learningRate', [1 2], ...
    'weightDecay', [1 0], ...
    'opts', {convOpts});

if opts.batchNormalization
    id = id + 1;
    out = size(net.layers{end}.filters,4);
    net.layers{end+1} = struct('type', 'bnorm','name', sprintf('bn%d',id),'weights', {{ones(out, 1, 'single'),...
        zeros(out, 1, 'single'), zeros(out, 2, 'single')}}, 'learningRate', [2 1 0.1], ...
        'weightDecay', [0 0]) ;
end

% ReLu 2
net.layers{end+1} = struct('type', 'relu') ;

% Pool 1
net.layers{end+1} = struct('type', 'pool', ...
    'method', 'max', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad', 0) ;

% Conv 3 - Common
net.layers{end+1} = struct('type', 'conv', ...
    'filters', 0.01/scal * randn(3,3,64, 64, 'single'), ...%3X3, 32
    'biases', init_bias*ones(1, 64, 'single'), ...
    'stride', 1, ... %1
    'pad', 1, ...
    'learningRate', [1 2], ...
    'weightDecay', [1 0], ...
    'opts', {convOpts});

if opts.batchNormalization
    id = id + 1;
    out = size(net.layers{end}.filters,4);
    net.layers{end+1} = struct('type', 'bnorm','name', sprintf('bn%d',id),'weights', {{ones(out, 1, 'single'),...
        zeros(out, 1, 'single'), zeros(out, 2, 'single')}}, 'learningRate', [2 1 0.1], ...
        'weightDecay', [0 0]) ;
end

% ReLu 3
net.layers{end+1} = struct('type', 'relu') ;

% Conv 4 - Common
net.layers{end+1} = struct('type', 'conv', ...
    'filters', 0.01/scal * randn(3,3,64, 128, 'single'), ...%3X3, 32
    'biases', init_bias*ones(1, 128, 'single'), ...
    'stride', 1, ... %1
    'pad', 1, ...
    'learningRate', [1 2], ...
    'weightDecay', [1 0], ...
    'opts', {convOpts});

if opts.batchNormalization
    id = id + 1;
    out = size(net.layers{end}.filters,4);
    net.layers{end+1} = struct('type', 'bnorm','name', sprintf('bn%d',id),'weights', {{ones(out, 1, 'single'),...
        zeros(out, 1, 'single'), zeros(out, 2, 'single')}}, 'learningRate', [2 1 0.1], ...
        'weightDecay', [0 0]) ;
end

% ReLu 4
net.layers{end+1} = struct('type', 'relu');

% Pool 2
net.layers{end+1} = struct('type', 'pool', ...
    'method', 'max', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad', 0) ;

% Conv 5 - Common
net.layers{end+1} = struct('type', 'conv', ...
    'filters', 0.01/scal * randn(3,3, 128, 128, 'single'), ... %old 5X5
    'biases', init_bias*ones(1, 128, 'single'), ...
    'stride', 1, ...
    'pad', 1, ...
    'learningRate', [1 2], ...
    'weightDecay', [1 0], ...
    'opts', {convOpts});

if opts.batchNormalization
    id = id + 1;
    out = size(net.layers{end}.filters,4);
    net.layers{end+1} = struct('type', 'bnorm','name', sprintf('bn%d',id),'weights', {{ones(out, 1, 'single'),...
        zeros(out, 1, 'single'), zeros(out, 2, 'single')}}, 'learningRate', [2 1 0.1], ...
        'weightDecay', [0 0]) ;
end

% ReLu 5
net.layers{end+1} = struct('type', 'relu') ;

% Conv 6 - Common
net.layers{end+1} = struct('type', 'conv', ...
    'filters', 0.01/scal * randn(3,3, 128, 128, 'single'), ... %remove if 5x5 above
    'biases', init_bias*ones(1, 128, 'single'), ...
    'stride', 1, ...
    'pad', 1, ...
    'learningRate', [1 2], ...
    'weightDecay', [1 0], ...
    'opts', {convOpts});

if opts.batchNormalization
    id = id + 1;
    out = size(net.layers{end}.filters,4);
    net.layers{end+1} = struct('type', 'bnorm','name', sprintf('bn%d',id),'weights', {{ones(out, 1, 'single'),...
        zeros(out, 1, 'single'), zeros(out, 2, 'single')}}, 'learningRate', [2 1 0.1], ...
        'weightDecay', [0 0]) ;
end

% ReLu 6
net.layers{end+1} = struct('type', 'relu') ;

% Conv 7 - Common 9X9
net.layers{end+1} = struct('type', 'conv', ...
    'filters', 0.01/scal * randn(9,9, 128, 256, 'single'), ...
    'biases', init_bias*ones(1, 256, 'single'), ...
    'stride', 1, ...%1
    'pad', 4, ...
    'learningRate', [1 2], ...
    'weightDecay', [1 0], ...
    'opts', {convOpts});

if opts.batchNormalization
    id = id + 1;
    out = size(net.layers{end}.filters,4);
    net.layers{end+1} = struct('type', 'bnorm','name', sprintf('bn%d',id),'weights', {{ones(out, 1, 'single'),...
        zeros(out, 1, 'single'), zeros(out, 2, 'single')}}, 'learningRate', [2 1 0.1], ...
        'weightDecay', [0 0]) ;
end

% ReLu 7
net.layers{end+1} = struct('type', 'relu') ;

% Conv 8 - Common
net.layers{end+1} = struct('type', 'conv', ...
    'filters', 0.01/scal * randn(9,9, 256, 512, 'single'), ...
    'biases', init_bias*ones(1, 512, 'single'), ...
    'stride', 1, ...%1
    'pad', 4, ...
    'learningRate', [1 2], ...
    'weightDecay', [1 0], ...
    'opts', {convOpts});

if opts.batchNormalization
    id = id + 1;
    out = size(net.layers{end}.filters,4);
    net.layers{end+1} = struct('type', 'bnorm','name', sprintf('bn%d',id),'weights', {{ones(out, 1, 'single'),...
        zeros(out, 1, 'single'), zeros(out, 2, 'single')}}, 'learningRate', [2 1 0.1], ...
        'weightDecay', [0 0]) ;
end

% ReLu 8
net.layers{end+1} = struct('type', 'relu') ;

% Conv 9 - Common
net.layers{end+1} = struct('type', 'conv', ...
    'filters', 0.01/scal * randn(1,1, 512, 256, 'single'), ...
    'biases', init_bias*ones(1, 256, 'single'), ...
    'stride', 1, ...%1
    'pad', 0, ...
    'learningRate', [1 2], ...
    'weightDecay', [1 0], ...
    'opts', {convOpts});

% ReLu 9
net.layers{end+1} = struct('type', 'relu') ;

% Conv 10 - Common
net.layers{end+1} = struct('type', 'conv', ...
    'filters', 0.01/scal * randn(1,1, 256, 256, 'single'), ...
    'biases', init_bias*ones(1, 256, 'single'), ...
    'stride', 1, ...%1
    'pad', 0, ...
    'learningRate', [1 2], ...
    'weightDecay', [1 0], ...
    'opts', {convOpts});

% ReLu 10
net.layers{end+1} = struct('type', 'relu') ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Convert old strcture network to dagnn
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%skip layer
skip_layer = opts.skip_layer;

%reference layer
ref_layer = net.layers(end).name;

%Add Loss layer
loss_cnt=1;
temp_opts = opts;
temp_opts.sharedW=0; %no shared weights
net = addMultiLoss(temp_opts,convOpts,scal,init_bias,net,ref_layer,loss_cnt,[1,2,3]);

%%%%%%%%%%%%%%%%%%%%%%%SEQUENTIAL PART - END%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%RECURRENT PART - START%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lossIdx = [4,5,6];
for i=1:Niter %this iterator is the ref_idx later
    opts.sharedW = lopts.shareFlag(i); %shared weights flag
    %the first iteration has never shared weights
    
    net = addRecFuse(opts,net,scal,init_bias,skip_layer,ref_layer,i,lopts.shareFlag(i), lopts.ref_idx, resConn);
    
    %new reference layer
    ref_layer = net.layers(end).outputs;
    
    %Add Loss layers
    loss_cnt=loss_cnt+1;
    net = addMultiLoss(opts,convOpts,scal,init_bias,net,ref_layer,loss_cnt,lossIdx);
    
    lossIdx = lossIdx + numel(lossIdx);
end

end

function net = addMultiLoss(opts,convOpts,scal,init_bias,net,ref_layer,idx,idxloss)
% Active Ojbectives: keypoints and body parts.
% The human body objective has been removed.

x = net.getLayerIndex(ref_layer);
ref_layer = net.layers(x).outputs;

if ~isempty(opts.lossFunc)
    
    %Conv A - Keypoints
    filt_size = [1 1 256 opts.outNode];
    convBlock11 = dagnn.Conv('size',filt_size, 'pad', [0,0,0,0],'stride', [1,1], ...
        'hasBias', true, 'opts', convOpts);
    
    if ~opts.sharedW
        net.addLayer(sprintf('conv%d_key',idx), convBlock11, {ref_layer{1}}, {sprintf('conv%d_key',idx)}, {sprintf('conv%d_A_filters',idx),sprintf('conv%d_A_biases',idx)});
        
        f = net.getParamIndex(sprintf('conv%d_A_filters',idx)) ;
        net.params(f).value = 0.01/scal.*randn(filt_size, 'single') ;
        net.params(f).learningRate=1;
        net.params(f).weightDecay=1;
        
        f = net.getParamIndex(sprintf('conv%d_A_biases',idx)) ;
        net.params(f).value = init_bias*ones(1,opts.outNode, 'single');
        net.params(f).learningRate=2;
        net.params(f).weightDecay=0;
    else
        ref_idx=2;
        net.addLayer(sprintf('conv%d_key',idx), convBlock11, {ref_layer{1}}, {sprintf('conv%d_key',idx)}, {sprintf('conv%d_A_filters',ref_idx),sprintf('conv%d_A_biases',ref_idx)});
    end
    
    %ReLu 11A
    net.addLayer(sprintf('relu%d_A',idx), dagnn.ReLU(), {sprintf('conv%d_key',idx)}, {sprintf('prediction%d',idxloss(1))}, {}) ;
    
    %Objective1 - keypoints heatmaps
    net.addLayer(sprintf('objective%d',idxloss(1)), dagnn.RegLoss('loss', opts.lossFunc), ...
        {sprintf('prediction%d',idxloss(1)),'label'}, sprintf('objective%d',idxloss(1))) ;
    
    net.addLayer(sprintf('error%d',idxloss(1)), dagnn.RegLoss('loss', 'mse-heatmap'), ...
        {sprintf('prediction%d',idxloss(1)),'label'},sprintf('error%d',idxloss(1)));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%KEYPOINT HEATMAPS END%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isempty(opts.lossFunc2)
    
    %Conv 11B - Body parts
    filt_size = [1 1 256 opts.outPairNode];
    convBlock11 = dagnn.Conv('size',filt_size, 'pad', [0,0,0,0],'stride', [1,1], ...
        'hasBias', true, 'opts', convOpts);
    
    if ~opts.sharedW
        net.addLayer(sprintf('conv%d_part',idx), convBlock11, {ref_layer{1}}, {sprintf('conv%d_part',idx)}, {sprintf('conv%d_B_filters',idx),sprintf('conv%d_B_biases',idx)});
        
        f = net.getParamIndex(sprintf('conv%d_B_filters',idx)) ;
        net.params(f).value = 0.01/scal.*randn(filt_size, 'single') ;
        net.params(f).learningRate=1;
        net.params(f).weightDecay=1;
        
        f = net.getParamIndex(sprintf('conv%d_B_biases',idx)) ;
        net.params(f).value = init_bias*ones(1,opts.outPairNode, 'single');
        net.params(f).learningRate=2;
        net.params(f).weightDecay=0;
    else
        ref_idx=2;
        net.addLayer(sprintf('conv%d_part',idx), convBlock11, {ref_layer{1}}, {sprintf('conv%d_part',idx)}, {sprintf('conv%d_B_filters',ref_idx),sprintf('conv%d_B_biases',ref_idx)});
    end
    
    %ReLu 11B
    net.addLayer(sprintf('relu%d_B',idx), dagnn.ReLU(), {sprintf('conv%d_part',idx)}, {sprintf('prediction%d',idxloss(2))}, {}) ;
    
    %Objective2 - keypoints heatmaps
    net.addLayer(sprintf('objective%d',idxloss(2)), dagnn.RegLoss('loss', opts.lossFunc2), ...
        {sprintf('prediction%d',idxloss(2)),'label'}, sprintf('objective%d',idxloss(2))) ;
    
    net.addLayer(sprintf('error%d',idxloss(2)), dagnn.RegLoss('loss', 'mse-pairwiseheatmap'), ...
        {sprintf('prediction%d',idxloss(2)),'label'},sprintf('error%d',idxloss(2)));
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%BODY PART HEATMAPS END%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isempty(opts.lossFunc3)
    
    % skip this objective
end

end

function net = addRecFuse(opts,net,scal,init_bias,skip_layer,ref_layer,iter, sharedW, ref_idx, resConn)
% Recurrent Iteration (with or without shared weights)

opts.layerName = 'fu';

x = net.getLayerIndex(skip_layer);
skip_layer = net.layers(x).outputs;

x = net.getLayerIndex(ref_layer);
ref_layer = net.layers(x).outputs;

% Concatenate
net.addLayer(sprintf('skip_maps%d',iter), dagnn.Concat(), {skip_layer{1},ref_layer{1}}, {sprintf('skip_maps%d',iter)});

%Conv 12 + ReLu
dimos = [7 7 opts.ConcFeat 64];%768,384
pados = [3,3,3,3];
idx=1; %batch normalization
opts.batchNormalization=1;
opts.inputVar = net.layers(end).outputs{1};
net=addConv(opts, scal, init_bias, net, dimos, pados, iter, idx, sharedW, ref_idx, ref_layer, 0);

%Conv 13 + ReLu
dimos = [13 13 64 64];
pados = [6,6,6,6];
idx = idx + 1; %batch normalization
opts.batchNormalization=1;
opts.inputVar = net.layers(end).outputs{1};
net=addConv(opts, scal, init_bias, net, dimos, pados, iter, idx, sharedW, ref_idx, ref_layer, 0);

%Conv 14 + ReLu
dimos = [13 13 64 128];
pados = [6,6,6,6];
idx = idx + 1; %batch normalization
opts.batchNormalization=1;
opts.inputVar = net.layers(end).outputs{1};
net=addConv(opts, scal, init_bias, net, dimos, pados, iter, idx, sharedW, ref_idx, ref_layer, 0);

%Conv 14 + ReLu
dimos = [1 1 128 256];
pados = [0,0,0,0];
idx = idx + 1; %batch normalization
opts.batchNormalization=0; %no batch for 1x1
opts.inputVar = net.layers(end).outputs{1};
net=addConv(opts, scal, init_bias, net, dimos, pados, iter, idx, sharedW, ref_idx, ref_layer, resConn);

end

function net=addConv(opts, scal, init_bias, net, dimos, pados, iter, idx, sharedW, ref_idx, ref_layer, resConn)

%Conv + Bnrom + ReLu
convBlockFu = dagnn.Conv('size', dimos, 'pad', pados,'stride', [1,1], ...
    'hasBias', true);

if ~sharedW
    net.addLayer(sprintf('%s_%d_conv%d',opts.layerName,iter,idx), convBlockFu, {opts.inputVar}, {sprintf('%s_%d_conv%d',opts.layerName,iter,idx)}, {sprintf('%s_%d_%d_filters',opts.layerName,iter,idx), sprintf('%s_%d_%d_biases',opts.layerName,iter,idx)}) ;
    
    f = net.getParamIndex(sprintf('%s_%d_%d_filters',opts.layerName,iter,idx)) ;
    net.params(f).value = 0.01/scal.*randn(dimos, 'single') ;
    net.params(f).learningRate=1;
    net.params(f).weightDecay=1;
    
    f = net.getParamIndex(sprintf('%s_%d_%d_biases',opts.layerName,iter,idx)) ;
    net.params(f).value = init_bias*ones(1, dimos(4), 'single');
    net.params(f).learningRate=2;
    net.params(f).weightDecay=0;
else
    %use the weights from the reference iteration (usually the first one)
    net.addLayer(sprintf('%s_%d_conv%d',opts.layerName,iter,idx), convBlockFu, {net.layers(end).outputs{1}}, {sprintf('%s_%d_conv%d',opts.layerName,iter,idx)}, {sprintf('%s_%d_%d_filters',opts.layerName,ref_idx,idx), sprintf('%s_%d_%d_biases',opts.layerName,ref_idx,idx)}) ;
end

%Batch norm
in=sprintf('%s_%d_conv%d',opts.layerName,iter,idx);
if opts.batchNormalization
    out=dimos(4);
    params={sprintf('%s_%d_%d_bn_m',opts.layerName,iter,idx),sprintf('%s_%d_%d_bn_b',opts.layerName,iter,idx),sprintf('%s_%d_%d_bn_x',opts.layerName,iter,idx)};
    net.addLayer(sprintf('%s_%d_%d_bn',opts.layerName,iter,idx), dagnn.BatchNorm(), {in}, {sprintf('%s_%d_%d_bn',opts.layerName,iter,idx)},params) ;
    f = net.getParamIndex(sprintf('%s_%d_%d_bn_m',opts.layerName,iter,idx));
    net.params(f).value = ones(out, 1, 'single');
    net.params(f).learningRate=2;
    net.params(f).weightDecay=0;
    f = net.getParamIndex(sprintf('%s_%d_%d_bn_b',opts.layerName,iter,idx));
    net.params(f).value = zeros(out, 1, 'single');
    net.params(f).learningRate=1;
    net.params(f).weightDecay=0;
    f = net.getParamIndex(sprintf('%s_%d_%d_bn_x',opts.layerName,iter,idx));
    net.params(f).value = zeros(out, 2, 'single');
    net.params(f).learningRate=0.1;
    net.params(f).weightDecay=0;
    
    %Residual Connection (not used in the final experiments)
    if resConn
        resRef = net.layers(end).outputs;
        net.addLayer(sprintf('%s_res_sum%d',opts.layerName,iter), dagnn.Sum(), {ref_layer{1},resRef{1}},sprintf('%s_res_sum%d',opts.layerName,iter));
        net.addLayer(sprintf('relu_%s_%d_%d',opts.layerName,iter,idx),  dagnn.ReLU(), {sprintf('%s_res_sum%d',opts.layerName,iter)}, {sprintf('relu_%s_%d_%d',opts.layerName,iter,idx)}, {});
    else
        net.addLayer(sprintf('relu_%s_%d_%d',opts.layerName,iter,idx),  dagnn.ReLU(), {sprintf('%s_%d_%d_bn',opts.layerName,iter,idx)}, {sprintf('relu_%s_%d_%d',opts.layerName,iter,idx)}, {});
    end
else
    %Residual Connection (not used in the final experiments)
    if resConn
        resRef = net.layers(end).outputs;
        net.addLayer(sprintf('%s_res_sum%d',opts.layerName,iter), dagnn.Sum(), {ref_layer{1},resRef{1}},sprintf('%s_res_sum%d',opts.layerName,iter));
        net.addLayer(sprintf('relu_%s_%d_%d',opts.layerName,iter,idx),  dagnn.ReLU(), {sprintf('%s_res_sum%d',opts.layerName,iter)}, {sprintf('relu_%s_%d_%d',opts.layerName,iter,idx)}, {});
    else
        net.addLayer(sprintf('relu_%s_%d_%d',opts.layerName,iter,idx),  dagnn.ReLU(), {in}, {sprintf('relu_%s_%d_%d',opts.layerName,iter,idx)}, {});
    end
end
%Conv + Bnrom + ReLu

end