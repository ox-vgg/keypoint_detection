function Y = vl_nnheatmaploss(X, c, dzdy, varargin)
%VL_NNHEATMAPLOSS computes the loss over heatmap predictions
%   Y = VL_NNHEATMAPLOSS(X, C) 
%
%   VL_NNHEATMAPLOSS(..., 'option', value, ...) takes the following 
%   options:
%
%   `ignOcc`:: false
%    Ignore occluded keypoints in the loss evaluation 
%
% Copyright (C) 2016  Vasileios Belagiannis and Samuel Albanie
% All rights reserved.
% Contact: vb@robots.ox.ac.uk

  opts.ignOcc = false ;
  opts.loss = 'l2loss-heatmap' ;
  opts = vl_argparse(opts,varargin) ;

  switch lower(opts.loss)
    case {'l2loss-heatmap', 'l2loss-pairwiseheatmap'}
      %GT
      if strcmp(opts.loss,'l2loss-heatmap')
        if iscell(c)
          Y = cat(4,c{2,:}) ;
        else
          Y = c ;
          c = opts.labels ;
        end
        weight_mask = cat(4,c{3,:});
      elseif strcmp(opts.loss,'l2loss-pairwiseheatmap')
        if iscell(c)
          Y = cat(4,c{8,:});
        else
          Y = c;
          c = opts.labels;
        end
        weight_mask = cat(4,c{9,:});
      end
      res = (Y-X) ;
      %missing annotation - zeros contribution
      idx = repmat(sum(sum(Y,1),2)==0,size(res,1),size(res,2)) ;
      res(idx) = zeros(size(res(idx)), 'like', res(idx)) ; %check it again!!!
      n = 1 ;
      if isempty(dzdy) %forward
        scale = (size(res,1)*size(res,2)*size(res,3)) *1000 ;
        Y = sqrt(sum(res(:).^2)) / scale ; % normalize by scale factor
      else
        %if occluded keypoints - ignore them
        if opts.ignOcc
          idxOcc = Y < 0 ;
          res(idxOcc) = zeros(size(res(idxOcc)), 'like', res(idxOcc)) ;
        end
        %gradient weighting
        res = weight_mask .* res ;
        Y_= -1.*res ;
        Y = single (Y_ * (dzdy / n) ) ;
      end
    case {'mse-heatmap', 'mse-pairwiseheatmap'} % mean squarred error
      % GT stored in sparse matrices stacked next to each other
      if strcmp(opts.loss,'mse-heatmap')
        Y = cat(4,c{2,:});
      elseif strcmp(opts.loss,'mse-pairwiseheatmap')
        Y = cat(4,c{8,:});
      end
      if isempty(dzdy) %forward
        fun = @(A,B) A-B;
        err = bsxfun(fun,Y,X);
        %missing annotation - zeros contribution
        idx=repmat(sum(sum(Y,1),2)==0,size(err,1),size(err,2));
        err(idx) = zeros(size(err(idx)), 'like', err(idx)) ; %check it again!!!
        
        %if occluded keypoints - ignore them
        if opts.ignOcc
          idxOcc=Y<0 ;
          err(idxOcc) = zeros(size(err(idxOcc)), 'like', err(idxOcc)) ;
        end
        %dim 1 - 62, dim 2 - 62, dim 3 - 16 (body joints -heatmaps)
        Y = sum(err(:).^2)/(size(X,1)*size(X,2)*size(X,3));
      else, Y = zeros(size(X), 'like', X) ; %nothing to backprop
      end
    otherwise, error('Unknown loss ''%s''.', opts.loss) ;
  end
