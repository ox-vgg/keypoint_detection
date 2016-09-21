function [YA, YB] = vl_nnstructloss(XA, XB, c, dzdy, varargin)

opts.loss = 'strloss' ;
opts.iter=0;
opts.bodyPairs=[];
opts.trf=[];
opts.N_modes = 5;
opts = vl_argparse(opts,varargin) ;

marg = 30;

backFlg = 0;

YA=0;
YB=0;

XA = gather(XA);
XB = gather(XB);

n=0;

switch lower(opts.loss)
    case {'strloss'}
        XgtA = cat(4,c{2,:}); %keypoints
        XgtB = cat(4,c{8,:}); %parts
        
        %backwards
        if ~isempty(dzdy)
            DerA = zeros(size(XA,1),size(XA,2),size(XA,3));
            DerB = zeros(size(XB,1),size(XB,1),size(XB,3));
            [TempDerA{1:size(XA,4), 1}] = deal(DerA);
            [TempDerB{1:size(XB,4), 1}] = deal(DerB);
            backFlg=1;
        end
        
        for i=1:size(XA,4)
            
            %score for ground-truth pose (in the output space)
            poseGt = getGTscore(opts,XgtA(:,:,:,i),XgtB(:,:,:,i),c{1,i});
            poseGt = addPairScores (opts,XgtA(:,:,:,i),XgtB(:,:,:,i), poseGt);
            
            %score for best infered pose (in the output space)
            posePr = runLinProg(opts,XA(:,:,:,i),XB(:,:,:,i),opts.N_modes,backFlg);
            posePr = addPairScores (opts, XA(:,:,:,i), XB(:,:,:,i), posePr);
            posePr(logical(sum(poseGt')==0)',:) = 0;%remove missing anot.
           
            Score_y_gt = sum(poseGt(:,3));
            Score_y_pred = sum(posePr(:,3));
         
            score = (marg+Score_y_pred - Score_y_gt);
            
            if isempty(dzdy) %forward
                YA =  YA + sum(poseGt(1:end - size(opts.bodyPairs,2),3));
                YB =  YB + sum(poseGt(end - size(opts.bodyPairs,2)+1:end,3));
                
            else  %backward
               
                if score>0
                    %exclude predictions that are close to GT & keep the most violated
                    di = sqrt(((poseGt(:,1) - posePr(:,1)) + (poseGt(:,2) - posePr(:,2))).^2);
                    di = di<5;%threshold of correctness (ideally use PCP)
                    poseGt(di,:)=0;%remove the correct predictions
                    posePr(di,:)=0;%remove the GT for the correct predictions
                    
                    %Predictions - Keypoints
                    poseDer = posePr(1:end-size(opts.bodyPairs,2),1:2);
                    channelZ = find((sum(poseDer')~=0)');
                    poseDer = poseDer((sum(poseDer')~=0)',:);
                    %idx = [poseDer channelZ i*ones(size(poseDer,1),1)];
                    idx = [poseDer channelZ];
                    TempDerA{i}(idx) = 1;
                    
                    poseDer = posePr(end-size(opts.bodyPairs,2)+1:end,1:2);
                    channelZ = find((sum(poseDer')~=0)');
                    poseDer = poseDer((sum(poseDer')~=0)',:);
                    %idx = [poseDer channelZ i*ones(size(poseDer,1),1)];
                    idx = [poseDer channelZ];
                    TempDerB{i}(idx) = 1;
                    
                    %Ground-truth - Keypoints
                    poseDer = poseGt(1:end-size(opts.bodyPairs,2),1:2);
                    channelZ = find((sum(poseDer')~=0)');
                    poseDer = poseDer((sum(poseDer')~=0)',:);
                    %idx = [poseDer channelZ i*ones(size(poseDer,1),1)];
                    idx = [poseDer channelZ];
                    TempDerA{i}(idx) = -1;
                    
                    poseDer = poseGt(end-size(opts.bodyPairs,2)+1:end,1:2);
                    channelZ = find((sum(poseDer')~=0)');
                    poseDer = poseDer((sum(poseDer')~=0)',:);
                    %idx = [poseDer channelZ i*ones(size(poseDer,1),1)];
                    idx = [poseDer channelZ];
                    TempDerB{i}(idx) = -1;
                    
                    %else score(i)<=0 -> 0 (already from the init.)
                    
                    %sum up the contributing examples
                    n = n + sum(di);
                end
            end
            
        end
        
        if ~isempty(dzdy) %backward
            DerA = cat(4,TempDerA{:});%use for the parfor
            DerB = cat(4,TempDerB{:});
            YA = single(DerA./n);
            YB = single(DerB./n);
        end
        
    otherwise
        error('Unknown loss ''%s''.', opts.loss) ;
end

% --------------------------------------------------------------------
function y = zerosLike(x)
% --------------------------------------------------------------------
if isa(x,'gpuArray')
    y = gpuArray.zeros(size(x),'single') ;
else
    y = zeros(size(x),'single') ;
end

