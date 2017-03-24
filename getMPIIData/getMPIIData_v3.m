clearvars;close all; clc;

%Download MPII dataset
if ~exist('mpii_human_pose_v1.tar.gz', 'file')
url = 'http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz';
filename = 'mpii_human_pose_v1.tar.gz';
websave(filename,url);
untar(filename);
end

%Download MPII annotations
if ~exist('mpii_human_pose_v1_u12_2.zip', 'file')
url = 'http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip';
filename = 'mpii_human_pose_v1_u12_2.zip';
websave(filename,url);
unzip(filename);
end

%Download validation dataset from J. Tompshon
if ~exist('mpii_valid_pred.zip', 'file')
url = 'http://www.cims.nyu.edu/~tompson/data/mpii_valid_pred.zip';
filename = 'mpii_valid_pred.zip';
websave(filename,url);
unzip(filename);
end

load('mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1');
annolist_test = RELEASE.annolist(RELEASE.img_train == 0);
single_person_test = RELEASE.single_person(RELEASE.img_train == 0);
testMap = zeros(numel(single_person_test),2);
data = RELEASE;
clear RELEASE;
N = length(data.annolist);

%Validation data from Tompson
load('mpii_predictions/data/detections');

%set a flag for excluding validation poses the heatmap generation
for i=1:length(RELEASE_img_index)
    data.annolist(RELEASE_img_index(i)).valPerson=RELEASE_person_index(i);
end


%patch bugs in the dataset
data.annolist(24731).annorect(2).annopoints.point.is_visible=1;
for i=1:numel(data.annolist)
    for j=1:numel(data.annolist(i).annorect)
        if isfield(data.annolist(i).annorect(j),'annopoints') %GT exists
            if ~isempty(data.annolist(i).annorect(j).objpos) %localize objects
                for k=1:numel(data.annolist(i).annorect(j).annopoints.point)
                    if ischar(data.annolist(i).annorect(j).annopoints.point(k).is_visible)
                        num = str2double(data.annolist(i).annorect(j).annopoints.point(k).is_visible);
                        data.annolist(i).annorect(j).annopoints.point(k).is_visible=num;
                    end
                end
            end
        end
    end
end
%patch bugs in the dataset

%network image input size
imgSize=[256, 256];

%activate the human detector
detector=0;

%best detection input
oracleTr=0;

%base height (take a larger bounding box)
baseH = 300;

%base width (take a larger bounding box)
baseW = 300;

h = waitbar(0,'MPI dataset...');

%train samples
sets_train=zeros(length(data.img_train),1);

cnt=0;
cnt_test=0;
for i=1:length(data.img_train)
    
    if exist(['images/' data.annolist(i).image.name], 'file');
        img = imread(['images/' data.annolist(i).image.name]);
    else
        img=[];
    end
    
    %check test data
    %         if data.img_train(i) ==0 && ~isempty(data.single_person{i})
    %
    %             %imshow(img); disp(sprintf('train %d',data.img_train(i))); pause();
    %             imt = imresize(img,1/data.annolist(i).annorect(1).scale);
    %             subplot(2,1,1);
    %             imshow(imt); hold on;
    %             wid = 200;
    %             hei= 250;
    %             xUpLe=(data.annolist(i).annorect(1).objpos.x)/data.annolist(i).annorect(1).scale-(wid/2);
    %             yUpLe=(data.annolist(i).annorect(1).objpos.y)/data.annolist(i).annorect(1).scale-(hei/2);
    %
    %             %check if the bounding box exceeds the image plane
    %             xUpLe=round(max(1,xUpLe));
    %             wid=round(min(size(imt,2),xUpLe+wid-1)-xUpLe);
    %             yUpLe=round(max(1,yUpLe));
    %             hei=round(min(size(imt,1),yUpLe+hei-1)-yUpLe);
    %
    %             rectangle('Position',[xUpLe yUpLe wid hei],'EdgeColor', 'r','LineWidth',5);
    %             hold off;
    %
    %             subplot(2,1,2);
    %             imshow(img); hold on;
    %             wid=200*data.annolist(i).annorect(1).scale;
    %             hei=250*data.annolist(i).annorect(1).scale;
    %             xUpLe=(data.annolist(i).annorect(1).objpos.x)-(wid/2);
    %             yUpLe=(data.annolist(i).annorect(1).objpos.y)-(hei/2);
    %
    %             %check if the bounding box exceeds the image plane
    %             xUpLe=round(max(1,xUpLe));
    %             wid=round(min(size(img,2),xUpLe+wid-1)-xUpLe);
    %             yUpLe=round(max(1,yUpLe));
    %             hei=round(min(size(img,1),yUpLe+hei-1)-yUpLe);
    %
    %             rectangle('Position',[xUpLe yUpLe wid hei],'EdgeColor', 'r','LineWidth',5);
    %             hold off;
    %
    %             disp(sprintf('id: %d train %d',i, data.img_train(i))); pause();
    %         end
    
    if data.img_train(i) ==0 && numel(data.single_person{i})>0
        %for j=1:numel(data.annolist(i).annorect)
        for k=1:numel(data.single_person{i})
            j = data.single_person{i}(k);
            if isfield(data.annolist(i).annorect(j),'objpos') %GT exists
                if ~isempty(data.annolist(i).annorect(j).objpos) %localize objects
                    cnt_test = cnt_test + 1;
                    
                    hei=round(baseH*data.annolist(i).annorect(j).scale);
                    wid=round(baseW*data.annolist(i).annorect(j).scale);
                    
                    obj_pose(1) = data.annolist(i).annorect(j).objpos.x;
                    obj_pose(2) = data.annolist(i).annorect(j).objpos.y;
                    
                    xUpLe=round(obj_pose(1)-wid/2);
                    yUpLe=round(obj_pose(2)-hei/2);
                    
                    %check if the bounding box exceeds the image plane
                    %and pad the image and all GT poses
                    padUpX = ceil(abs(min(0,xUpLe)));
                    padUpY = ceil(abs(min(0,yUpLe)));
                    xUpLe = max(1,xUpLe + padUpX);
                    yUpLe = max(1,yUpLe + padUpY);
                    padLoX = ceil(max(size(img,2), xUpLe + wid) - size(img,2));
                    padLoY = ceil(max(size(img,1), yUpLe + hei) - size(img,1));
                    
                    imgPad = uint8(128*ones(padUpY+size(img,1)+padLoY, ...
                        padUpX+size(img,2)+padLoX,3));
                    imgPad(1+padUpY:padUpY+size(img,1),1+padUpX:padUpX+size(img,2),:) = img;
                    
                    %check if the bounding box exceeds the image plane
                    %xUpLe=round(max(1,xUpLe));
                    %wid=round(min(size(img,2),xUpLe+wid-1)-xUpLe);
                    %yUpLe=round(max(1,yUpLe));
                    %hei=round(min(size(img,1),yUpLe+hei-1)-yUpLe);
                    
                    bbox_test{cnt_test} = [xUpLe, yUpLe, wid, hei];
                    pad_test{cnt_test} = [padUpX, padUpY, padLoX, padLoY];
                    
                    %crop the image
                    img_final_test{cnt_test}=imgPad(yUpLe:yUpLe+hei,xUpLe:xUpLe+wid,:);
                    
                    %resize to standard size
                    s_s = [size(img_final_test{cnt_test},1) size(img_final_test{cnt_test},2)];
                    s_t = [imgSize(1) imgSize(2)];
                    s = s_s.\s_t;
                    
                    %image resized
                    img_final_test{cnt_test} = imresize(img_final_test{cnt_test}, 'scale', s, 'method', 'bilinear');
                    
                    %index for mapping back
                    testMap(cnt_test,1) = i;
                    testMap(cnt_test,2) = j;
                    
                    %imshow(img_final_test{cnt_test});disp(i); pause(); close;
                end
            end
        end
    end
    
    %for each instance
    for j=1:numel(data.annolist(i).annorect)
        
        %if isfield(data.annolist(i).annorect(j),'objpos')
        if isfield(data.annolist(i).annorect(j),'annopoints') %GT exists
            if ~isempty(data.annolist(i).annorect(j).objpos) %localize objects
                cnt=cnt+1;
                
                sets_train(i)=1; %include in training
                
                sets_train_idx(cnt,1)=i;
                sets_train_idx(cnt,2)=j;
                
                %store the joints of the active individual
                ptsAll{cnt}=zeros(16,3);
                poseGT = data.annolist(i).annorect(j).annopoints.point;
                for p=1:numel(poseGT)
                    ptsAll{cnt}(poseGT(p).id+1,1)=poseGT(p).x +1;%1-indexed
                    ptsAll{cnt}(poseGT(p).id+1,2)=poseGT(p).y +1;
                    if ~isempty(poseGT(p).is_visible)
                        ptsAll{cnt}(poseGT(p).id+1,3)=poseGT(p).is_visible;
                    else
                        ptsAll{cnt}(poseGT(p).id+1,3)=1;%head
                    end
                end
                
%                 %plot the keypoints
%                 for p=1:size(ptsAll{cnt},1)
%                     if (ptsAll{cnt}(p,1)==0 && ptsAll{cnt}(p,3)==1) || (ptsAll{cnt}(p,2)==0 && ptsAll{cnt}(p,3)==1)
%                         imshow(img); hold on;
%                         plot(ptsAll{cnt}(p,1),ptsAll{cnt}(p,2),'rx');
%                         hold off; pause();
%                     end
%                 end
                
                %store the joints of the rest individuals
                valPerson=0;
                if ~isempty(data.annolist(i).valPerson) %check for validation indiv.
                    valPerson=data.annolist(i).valPerson;
                end
                
                ptsRest{cnt,1}=[];%required initialization
                if valPerson==0 %multiple individuals only for training frames
                
                cnt_rest=0;
                for jrest=1:numel(data.annolist(i).annorect)
                    %exclude the active indiv. & validation indiv. (if any)
                    if jrest~=j && jrest~=valPerson && ...
                            ~isempty(data.annolist(i).annorect(jrest).annopoints) %missing annotation
                        cnt_rest=cnt_rest+1;
                        ptsRest{cnt,cnt_rest}=zeros(16,3);
                        poseGT = data.annolist(i).annorect(jrest).annopoints.point;
                        for p=1:numel(poseGT)
                            ptsRest{cnt,cnt_rest}(poseGT(p).id+1,1)=poseGT(p).x +1;%1-indexed;
                            ptsRest{cnt,cnt_rest}(poseGT(p).id+1,2)=poseGT(p).y +1;%1-indexed;
                            if ~isempty(poseGT(p).is_visible)
                                ptsRest{cnt,cnt_rest}(poseGT(p).id+1,3)=poseGT(p).is_visible;
                            else
                                ptsRest{cnt,cnt_rest}(poseGT(p).id+1,3)=1;%head
                            end
                        end
                    end
                end
                
                %debug
                %if jrest==valPerson
                %    disp(['validation individual should be excluded ' num2str(valPerson)]);
                %end
                
                end %if - valPerson==0
                
                hei=round(baseH*data.annolist(i).annorect(j).scale);
                wid=round(baseW*data.annolist(i).annorect(j).scale);
                
                obj_pose(1) = data.annolist(i).annorect(j).objpos.x;
                obj_pose(2) = data.annolist(i).annorect(j).objpos.y;
                
                %imshow(img); hold on;
                %text(obj_pose(1),obj_pose(2),'C','Color','m','FontSize',22);
                %pause(); hold off;
                
                xUpLe=round(obj_pose(1)-wid/2);
                yUpLe=round(obj_pose(2)-hei/2);
                
                %check if the bounding box exceeds the image plane
                %and pad the image and all GT poses
                padUpX = ceil(abs(min(0,xUpLe)));
                padUpY = ceil(abs(min(0,yUpLe)));
                xUpLe = max(1,xUpLe + padUpX);
                yUpLe = max(1,yUpLe + padUpY);
                padLoX = ceil(max(size(img,2), xUpLe + wid) - size(img,2));
                padLoY = ceil(max(size(img,1), yUpLe + hei) - size(img,1));
                
                imgPad = uint8(128*ones(padUpY+size(img,1)+padLoY, ...
                    padUpX+size(img,2)+padLoX,3));
                imgPad(1+padUpY:padUpY+size(img,1),1+padUpX:padUpX+size(img,2),:) = img;
                
                %check if the bounding box exceeds the image plane
                %xUpLe=round(max(1,xUpLe));
                %wid=round(min(size(img,2),xUpLe+wid-1)-xUpLe);
                %yUpLe=round(max(1,yUpLe));
                %hei=round(min(size(img,1),yUpLe+hei-1)-yUpLe);
                
                bbox{cnt} = [xUpLe, yUpLe, wid, hei];
                pad_train{cnt} = [padUpX, padUpY, padLoX, padLoY];
                
                %crop the image
                img_final{cnt}=imgPad(yUpLe:yUpLe+hei,xUpLe:xUpLe+wid,:);
                
                %change the origin for the padded image
                idx=(ptsAll{cnt}(:,1)>0 & ptsAll{cnt}(:,2)>0);
                ptsAll{cnt}(idx,1)=ptsAll{cnt}(idx,1)+padUpX;
                ptsAll{cnt}(idx,2)=ptsAll{cnt}(idx,2)+padUpY;
                for jRest=1:size(ptsRest,2)
                    if ~isempty(ptsRest{cnt,jRest})
                        idx=(ptsAll{cnt}(:,1)>0 & ptsAll{cnt}(:,2)>0);
                        ptsRest{cnt,jRest}(idx,1)=ptsRest{cnt,jRest}(idx,1)+padUpX;
                        ptsRest{cnt,jRest}(idx,2)=ptsRest{cnt,jRest}(idx,2)+padUpY;
                    end
                end
                
                %shift the origin for the active individual
                ptsAll{cnt}(:,1)=ptsAll{cnt}(:,1)-(xUpLe-1);
                ptsAll{cnt}(:,2)=ptsAll{cnt}(:,2)-(yUpLe-1);
                checkX=double(ptsAll{cnt}(:,1)>0);
                checkY=double(ptsAll{cnt}(:,2)>0);
                ptsAll{cnt}(:,1:2)= ptsAll{cnt}(:,1:2).*[checkX checkX];
                ptsAll{cnt}(:,1:2)= ptsAll{cnt}(:,1:2).*[checkY checkY];
                checkX=double(ptsAll{cnt}(:,1)<=size(img_final{cnt},2));
                checkY=double(ptsAll{cnt}(:,2)<=size(img_final{cnt},1));
                ptsAll{cnt}(:,1:2)= ptsAll{cnt}(:,1:2).*[checkX checkX];
                ptsAll{cnt}(:,1:2)= ptsAll{cnt}(:,1:2).*[checkY checkY];
                
                %resize to standard size
                s_s = [size(img_final{cnt},1) size(img_final{cnt},2)];
                s_t = [imgSize(1) imgSize(2)];
                s = s_s.\s_t;
                tf = [ s(2) 0 0; 0 s(1) 0; 0  0 1];
                T = affine2d(tf);
                
                %points scaled
                [ptsAll{cnt}(:,1),ptsAll{cnt}(:,2)] = transformPointsForward(T, ptsAll{cnt}(:,1),ptsAll{cnt}(:,2));
                
                %shift the origin for the rest
                for jRest=1:size(ptsRest,2)
                    if ~isempty(ptsRest{cnt,jRest})
                        ptsRest{cnt,jRest}(:,1)=ptsRest{cnt,jRest}(:,1)-(xUpLe-1);
                        ptsRest{cnt,jRest}(:,2)=ptsRest{cnt,jRest}(:,2)-(yUpLe-1);
                        checkX=double(ptsRest{cnt,jRest}(:,1)>0);
                        checkY=double(ptsRest{cnt,jRest}(:,2)>0);
                        ptsRest{cnt,jRest}(:,1:2)=ptsRest{cnt,jRest}(:,1:2).*[checkX checkX];
                        ptsRest{cnt,jRest}(:,1:2)=ptsRest{cnt,jRest}(:,1:2).*[checkY checkY];
                        checkX=double(ptsRest{cnt,jRest}(:,1)<=size(img_final{cnt},2));
                        checkY=double(ptsRest{cnt,jRest}(:,2)<=size(img_final{cnt},1));
                        ptsRest{cnt,jRest}(:,1:2)=ptsRest{cnt,jRest}(:,1:2).*[checkX checkX];
                        ptsRest{cnt,jRest}(:,1:2)=ptsRest{cnt,jRest}(:,1:2).*[checkY checkY];
                        
                        %points scaled
                        idx=ptsRest{cnt,jRest}(:,1)>0;%not need for checking ptsRest{cnt,jRest}(:,2)>0;
                        [ptsRest{cnt,jRest}(idx,1),ptsRest{cnt,jRest}(idx,2)] ...
                            = transformPointsForward(T,ptsRest{cnt,jRest}(idx,1),ptsRest{cnt,jRest}(idx,2));
                        
                    end
                end
                clear xUpLe yUpLe wid hei padUpX padUpY padLoX padLoY;
                
                %image resized
                img_final{cnt} = imresize(img_final{cnt}, 'scale', s, 'method', 'bilinear');
                
                
%                 %visualization
%                 if valPerson~=0
%                     imshow(img_final{cnt}); hold on;
%                     x=size(img_final{cnt},2)/2;
%                     y=size(img_final{cnt},1)/2;
%                     text(x,y,'C','Color','m','FontSize',22);
%                     poseGT=ptsAll{cnt};
%                     for jj=1:1:size(poseGT,1) %active indiv.
%                         if poseGT(jj,3)==1
%                             text(poseGT(jj,1),poseGT(jj,2),int2str(jj),'Color','m','FontSize',16);
%                         else
%                             text(poseGT(jj,1),poseGT(jj,2),int2str(jj),'Color','r','FontSize',16);
%                         end
%                     end
%                     
%                     for jRest=1:size(ptsRest,2) %rest indiv.
%                         if ~isempty(ptsRest{cnt,jRest})
%                             poseGT=(ptsRest{cnt,jRest});
%                             for jj=1:1:size(poseGT,1)
%                                 if poseGT(jj,3)==1
%                                     text(poseGT(jj,1),poseGT(jj,2),int2str(jj),'Color','g','FontSize',16);
%                                 else
%                                     text(poseGT(jj,1),poseGT(jj,2),int2str(jj),'Color','r','FontSize',16);
%                                 end
%                             end
%                             
%                         end
%                     end
%                     pause();
%                     hold off;
%                     
%                     %check mapping back
%                     close all;
%                     imshow(img); hold on;
%                     %original GT points
%                     poseGT = data.annolist(i).annorect(j).annopoints.point;
%                     for p=1:numel(poseGT)
%                         text(poseGT(p).x,poseGT(p).y,int2str(poseGT(p).id+1),'Color','c','FontSize',16);
%                     end
%                 
%                     %transformed-points
%                     poseGT=ptsAll{cnt};
%                     [poseGT(:,1),poseGT(:,2)] = transformPointsInverse(T, ptsAll{cnt}(:,1),ptsAll{cnt}(:,2));
%                     poseGT(:,1) = poseGT(:,1) + bbox{cnt}(1)-1;
%                     poseGT(:,2) = poseGT(:,2) + bbox{cnt}(2)-1;
%                     poseGT(:,1) = poseGT(:,1) - pad_train{cnt}(1);
%                     poseGT(:,2) = poseGT(:,2) - pad_train{cnt}(2);
%                     for jj=1:1:size(poseGT,1) %active indiv.
%                         if poseGT(jj,3)==1
%                             text(poseGT(jj,1),poseGT(jj,2),int2str(jj),'Color','m','FontSize',16);
%                         else
%                             text(poseGT(jj,1),poseGT(jj,2),int2str(jj),'Color','r','FontSize',16);
%                         end
%                     end
%                    
%                     pause();
%                     hold off;
%                     
%                 end
%                 %visualization
                
            end
        end
    end
    
    %disp(i);
    waitbar(i / length(data.img_train));
end
close(h);


%storefile=sprintf('extractedData_detector_%d_orcacle_%d',detector, oracleTr);
storefile=sprintf('extractedData_%d_%d',imgSize(1),imgSize(2));
save(storefile,'img_final','ptsAll','ptsRest','sets_train','sets_train_idx','bbox','pad_train','-v7.3');

clear img_final bbox;
imgPath = img_final_test;
bbox = bbox_test;
poseGT = [];
storefile=sprintf('testMPI_%d_%d',imgSize(1),imgSize(2));
save(storefile,'imgPath','poseGT','bbox','testMap','pad_test','-v7.3');