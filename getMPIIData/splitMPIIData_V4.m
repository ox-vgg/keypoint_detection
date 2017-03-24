clearvars; close all; clc;

%load('extractedData_detector_0_orcacle_0.mat');
imgSize=[256, 256];
load(sprintf('extractedData_%d_%d',imgSize(1),imgSize(2)));

%validation data from Tompson
load('mpii_predictions/data/detections');

h = waitbar(0,'Please wait...');

ptsAll_train=[];
ptsAll_train_box=[];
imgPath_train=[];
ptsAll_test=[];
ptsAll_test_box=[];
imgPath_test=[];
tompson_val=[];

idx=1:16;

tompson_cnt=0; %counter for the validation data of Tompson
for i=1:size(sets_train_idx,1)
        
    cnt=1;
    indiv=1;
    poseGT(:,:,indiv)=ptsAll{i}(idx,:);
    while ~isempty(ptsRest{i,cnt}) && cnt<size(ptsRest,2)
        if sum(sum(ptsRest{i,cnt}(:,1:2)))>0
            indiv=indiv+1;
            poseGT(:,:,indiv)=ptsRest{i,cnt}(idx,:);
        end
        cnt=cnt+1;
    end
    
    idxImg=find(RELEASE_img_index==sets_train_idx(i,1));
    idxPe=find(RELEASE_person_index(idxImg)==sets_train_idx(i,2));
    if isempty(idxPe)
        
        ptsAll_train{1,size(ptsAll_train,2)+1}=poseGT;
      
        ptsAll_train_box(:,:,numel(ptsAll_train))=bbox{i};
        imgPath_train{numel(imgPath_train)+1}=img_final{i};
    else
        
        ptsAll_test{1,size(ptsAll_test,2)+1}=poseGT;
        
        ptsAll_test_box(:,:,numel(ptsAll_test))=bbox{i};
        imgPath_test{numel(imgPath_test)+1}=img_final{i};
        
        tompson_val(size(tompson_val,1)+1,1:2)=sets_train_idx(i,:);
        tompson_val(size(tompson_val,1),3:6)=bbox{i}; %bounding box for going back to the original coordinate system
        tompson_val(size(tompson_val,1),7:10)=pad_train{i};
    end
    
%     %plot the points
%     imshow(img_final{i}); hold on;
%     for j=1:size(poseGT,3)
%         tempY =poseGT(:,:,j);
%         for po=1:size(tempY,1)
%         text(tempY(po,1),tempY(po,2), int2str(po),'Color','m','FontSize',15);
%         end
%     end
%     hold off; pause();
    
    
    clear poseGT;
    
    waitbar(i / size(sets_train_idx,1));
end
close all;

clear ptsAll;


ptsAll=ptsAll_train;
imgPath=imgPath_train;

% for cnt=1:numel(imgPath)
%     %plot the points
%     imshow(imgPath{cnt}); hold on;
%     tempY = ptsAll(:,:,cnt);
%     for po=1:size(tempY,1)
%         text(tempY(po,1),tempY(po,2), int2str(po),'Color','m','FontSize',15);
%     end
%     hold off; pause();
%
% end

if numel(idx)==16
    save('MPI_imdbsT1aug0.mat','imgPath','ptsAll','-v7.3'); %pose data
else
    %do nothing
end

clear ptsAll imgPath;

ptsAll=ptsAll_test;
imgPath=imgPath_test;

if numel(idx)==16
    save('MPI_imdbsV1aug0.mat','imgPath','ptsAll','tompson_val','-v7.3'); %pose data
else
    %do nothing
end

close(h);