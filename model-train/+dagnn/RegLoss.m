classdef RegLoss < dagnn.ElementWise
    %Created by Vasileios Belagiannis.
    %Contact: vb@robots.ox.ac.uk
    
    properties
        loss = '' %default loss
        lossWeight = 1;
        ignOcc = 0;
        symm = 0;
    end
    
    properties (Transient)
        average = 0
        numAveraged = 0
        iter = 0
        scbox= 0
        hardNeg = 0
        vis = 0
        symmP= 0
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} = vl_nnheatmaploss(inputs{1}, inputs{2}, [], ...
                             'loss', obj.loss, 'ignOcc', obj.ignOcc) ;
            
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = vl_nnheatmaploss(inputs{1}, inputs{2}, ...
                    derOutputs{1}, 'loss', obj.loss, 'ignOcc', obj.ignOcc);
            derInputs{2} = [] ;
            derParams = {} ;
        end
        
        function reset(obj)
            obj.average = 0 ;
            obj.numAveraged = 0 ;
            obj.iter = 0 ;
            obj.scbox = 0 ;
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
            outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
        end
        
        function rfs = getReceptiveFields(obj)
            % the receptive field depends on the dimension of the variables
            % which is not known until the network is run
            rfs(1,1).size = [NaN NaN] ;
            rfs(1,1).stride = [NaN NaN] ;
            rfs(1,1).offset = [NaN NaN] ;
            rfs(2,1) = rfs(1,1) ;
        end
        
        function obj = RegLoss(varargin)
            obj.load(varargin) ;
        end
    end
end
