function Y = vl_nntukeyloss(X,c,iter,scbox, dzdy, varargin)

%Created by Vasileios Belagiannis.
%Contact: vb@robots.ox.ac.uk
%Only the l2loss for the heatmaps is included. The tukeyloss is in the
%following repo: https://github.com/bazilas/matconvnet-deepReg

opts.loss = 'l2loss-heatmap' ;
opts.lossWeight=1;
opts.ignOcc=0;
opts = vl_argparse(opts,varargin) ;

switch lower(opts.loss)
    %l2loss
    case {'l2loss'}
       
    case {'l2loss-heatmap'}
        if strcmp(opts.loss,'l2loss-heatmap')
            if iscell(c)
            Y = cat(4,c{2,:});
            else
            Y = c;
            c = opts.labels;
            end
            weight_mask = cat(4,c{3,:});
        end
        
        res=(Y-X);
        
        %missing annotation - zeros contribution 
        idx=repmat(sum(sum(Y,1),2)==0,size(res,1),size(res,2));
        res(idx)= zerosLike(res(idx)); %check it again!!!
        
        %n=sqrt(sum(res(:))); %L2 with square root
        n=1;
        if isempty(dzdy) %forward
            Y = sqrt(sum(res(:).^2))/(size(res,1)*size(res,2)*size(res,3)) *1000;%scale factor
        else
            %occluded keypoints - ignore them
            if opts.ignOcc
            idxOcc=Y<0;
            res(idxOcc)= zerosLike(res(idxOcc));
            end
                        
            res=weight_mask.*res;
            Y_= -1.*res;
            Y = single (Y_ * (dzdy / n) );
        end
        
    %error layer
    case {'mpe'} %mean pixel error
        
    case {'mae-heatmap'} %mean absolute error
   
    case {'mse-heatmap'} %mean squarred error
        %GT stored in sparse matrices stacked next to each other
        if strcmp(opts.loss,'mse-heatmap')
            Y = cat(4,c{2,:});
        end
        
        if isempty(dzdy) %forward
            
            fun = @(A,B) A-B;
            err = bsxfun(fun,Y,X);
            
            %missing annotation - zeros contribution
            idx=repmat(sum(sum(Y,1),2)==0,size(err,1),size(err,2));
            err(idx)= zerosLike(err(idx)); %check it again!!!
            
            %occluded keypoints - ignore them
            if opts.ignOcc
            idxOcc=Y<0;
            err(idxOcc)= zerosLike(err(idxOcc));
            end
            
            Y = sum(err(:).^2)/(size(X,1)*size(X,2)*size(X,3));%error per batch / not per samples
            
        else %nothing to backprop
            Y = zerosLike(X) ;
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

