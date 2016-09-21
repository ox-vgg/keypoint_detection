classdef StructLoss < dagnn.ElementWise
    properties
    loss = 'strloss' %default
    lossWeight = 1;
  end

  properties (Transient)
    average = 0
    numAveraged = 0
    iter = 0
    scbox= 0
    bodyPairs = []; %pairwise terms
    trf = []; %transformation from the output to the input space
  end
    
  methods
    function outputs = forward(obj, inputs, params)
      
      obj.iter = obj.net.iter;
      obj.scbox = obj.net.scbox;
      obj.bodyPairs = obj.net.bodyPairs;
      obj.trf = obj.net.trf;
      
      [outputs{1},outputs{2}] = vl_nnstructloss(inputs{1}, inputs{2}, inputs{3}, [], 'loss', obj.loss,'iter', obj.iter, 'bodyPairs', obj.bodyPairs, 'trf', obj.trf);
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1}) + gather(outputs{2})) / m;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      
      obj.iter = obj.net.iter;
      obj.scbox = obj.net.scbox;
      obj.bodyPairs = obj.net.bodyPairs;
      obj.trf = obj.net.trf;
       
      [derInputs{1},derInputs{2}] = vl_nnstructloss(inputs{1}, inputs{2}, inputs{3}, derOutputs{1}, 'loss', obj.loss,'iter', obj.iter, 'bodyPairs', obj.bodyPairs, 'trf', obj.trf);
      derInputs{3} = [] ;
      derParams = {} ;
    end
    
     function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
      obj.iter = 0 ;
      obj.scbox = 0 ;
    end
    
    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(1)] ;
    end
    
    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = StructLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
