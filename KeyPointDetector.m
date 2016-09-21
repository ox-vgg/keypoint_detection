classdef KeyPointDetector<handle
  properties
    norm_img_size = [248, 248];
    net_output_size = [62, 62];
    keypoint_names  = {'Right Ankle', 'Right Knee', 'Right Hip', 'Left Hip', 'Left Knee', ...
      'Left Ankle', 'Torso', 'Neck', 'Lower Head', 'Upper Head', 'Right Wirst', ...
      'Right Elbow', 'Right Shoulder', 'Left Shoulder', 'Left Elbow', 'Left Wirst'};
    gpu_id = -1;
    
    % supporting libraries
    matconvnet_dir = '';
    
    % Model
    model_name = 'keypoint_detector'
    model_fn = '';
    net = [];
    model_version = '';
  end
  
  methods
    
    function obj=KeyPointDetector(model_fn, matconvnet_dir, gpu_id)
      obj.model_fn = model_fn;
      obj.matconvnet_dir = matconvnet_dir;
      obj.gpu_id = gpu_id;
      
      % extract version number from model filename
      ver_match = regexp(model_fn, 'v\d+.mat', 'match');
      if length(ver_match) == 1
        obj.model_version = ver_match{1}(2:end-4);
      else
        obj.model_version = 'NA';
      end
      
      % load matconvnet into environment
      matconvnet_setup_fn = fullfile(matconvnet_dir, 'matlab', 'vl_setupnn.m');
      run(matconvnet_setup_fn);
      
      % Load model
      load(model_fn);
      obj.net = dagnn.DagNN.loadobj(net) ;
      
      % move the model to gpu
      if ( obj.gpu_id ~= 0 )
        gpuDevice(gpu_id);
        obj.net.move('gpu');
      end
      
    end
    
    function [kpx, kpy, kpname] = get_all_keypoints(obj, img_fn)
      img  = imread(img_fn);
      
      % Pad the image with zeros to make its square shaped
      diff = round((size(img,1)-size(img,2))/2);
      if diff>0 %pad width
        sq_img = padarray(img,[0,diff]);
      else
        sq_img = padarray(img,[-diff,0]);
      end
      [sq_imdim1, sq_imdim2, ~] = size(sq_img);
      
      % scale the image to a standard size
      sq_norm_img = imresize(sq_img, obj.norm_img_size);
      
      %single format and mean subtraction
      sq_norm_img = single(sq_norm_img);
      
      if ( obj.gpu_id ~= 0 )
        sq_norm_img = gpuArray(sq_norm_img);
      end
      
      sq_norm_img = bsxfun(@minus, sq_norm_img, single(repmat(128,1,1,3))) ; %subtract mean
      sq_norm_img = sq_norm_img./256;
      
      %evaluate the image
      obj.net.mode = 'test';
      obj.net.eval({'input', sq_norm_img}) ;
      net_output = obj.net.vars(obj.net.getVarIndex('prediction10')).value ;
      heatmap_count = size(net_output, 3);
      
      kpx = zeros(1, heatmap_count);
      kpy = zeros(1, heatmap_count);
      kpname = cell(1, heatmap_count);
      
      for hid=1:heatmap_count
        hmap = net_output(:, :, hid);
        [hy, hx, ~] = find( hmap == max(hmap(:)));
        sq_norm_img_x = (hx(1)/obj.net_output_size(1))*obj.norm_img_size(1);
        sq_norm_img_y = (hy(1)/obj.net_output_size(2))*obj.norm_img_size(2);
        
        sq_img_x = (sq_norm_img_x/obj.norm_img_size(1)) * sq_imdim1;
        sq_img_y = (sq_norm_img_y/obj.norm_img_size(1)) * sq_imdim2;
        
        img_x = sq_img_x;
        img_y = sq_img_y;
        
        if diff>0
          img_x = img_x - diff;
        else
          img_y = img_y + diff;
        end
        
        kpx(1,hid) = int32( gather( img_x ) );
        kpy(1,hid) = int32( gather( img_y ) );
        kpname{hid} = obj.keypoint_names{hid};
      end
    end
    
  end
end
