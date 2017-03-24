function [imo,pts] = imScaleGetBatch(im,pts)

% imshow(uint8(im));hold on;
% for i=1:size(pts,1)
%     text(pts(i,1),pts(i,2), 'x','Color','g','FontSize',15);
% end
% hold off; pause();

%scale from 0.7 to 1.3
s = 0.7 + rand(1)*(1.3-0.7);

tf = [s(1) 0 0; 0 s(1) 0; 0  0 1];
T = affine2d(tf);
[ptsx,ptsy] = transformPointsForward(T, pts(:,1),pts(:,2));
imo = imresize(im, 'scale', s, 'method', 'bilinear');

% imshow(uint8(imo));hold on;
% for i=1:size(pts,1)
%     text(ptsx(i),ptsy(i), 'x','Color','g','FontSize',15);
% end
% hold off;pause();

if size(imo,2) < size(im,2) %pad
    padsize = round((size(im,2) - size(imo,2))/2);
    imo = padarray(imo,[padsize padsize],128);
    %crop if it is required to euqalize the dims.
    imo=imo(1:size(im,1),1:size(im,2),:);
    ptsx=ptsx+padsize;
    ptsy=ptsy+padsize;
elseif size(imo,2) > size(im,2) %crop
    cropSize = round((size(imo,2) - size(im,1))/2);
    imo = imo(cropSize:end-cropSize,cropSize:end-cropSize,:);
    %crop if it is required to euqalize the dims.
    imo=imo(1:size(im,1),1:size(im,2),:);
    ptsx=ptsx-cropSize;
    ptsy=ptsy-cropSize;
end

if size(imo)~=size(im)
    disp('problem');
end

%zeros should remain zeros
ptsx = ptsx.*double(pts(:,1)>0 & pts(:,2)>0);
ptsy = ptsy.*double(pts(:,1)>0 & pts(:,2)>0);

%exclude out of plane points
ptsy = ptsy.*double(ptsx>=1 & ptsy>=1);
ptsx = ptsx.*double(ptsx>=1 & ptsy>=1);
ptsy = ptsy.*double(ptsy<size(imo,1) &  ptsx<size(imo,2));
ptsx = ptsx.*double(ptsy<size(imo,1) &  ptsx<size(imo,2));

clear pts;
pts=[ptsx ptsy];

% imshow(uint8(imo));hold on;
% for i=1:size(pts,1)
%     text(pts(i,1),pts(i,2), 'x','Color','g','FontSize',15);
% end
% hold off;pause();