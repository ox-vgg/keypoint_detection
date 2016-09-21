classdef DiffHeat < dagnn.ElementWise
 % Difference between the prediction and GT heatmap values
 
  properties
      labelGT=2;
  end

  properties (Transient)
    numInputs
  end

  methods
    function outputs = forward(obj, inputs, params)
      %input: 1.keypoint heatmap, 2. part heatmap, 3. human heatmap
      %input: 4.labels
      
      %transform the labels to tensors and subtract to get dx
      outputs{1} = cat(4,inputs{2}{obj.labelGT,:})-inputs{1};  
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = {} ;
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
    
    end

    function rfs = getReceptiveFields(obj)
      numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
      rfs = repmat(rfs, numInputs, 1) ;
    end

    function obj = DiffHeat(varargin)
      obj.load(varargin) ;
    end
  end
end
