function [imo,pts] = imRotateGetBatch(im,pts)

% imshow(uint8(im));hold on;
% for i=1:size(pts,1)
%     text(pts(i,1),pts(i,2), 'x','Color','g','FontSize',15);
% end
% hold off; pause();

theta1=rand(1)*40;
theta2=-theta1;

if rand(1)>0.5
    theta1=theta2;
end

tform = maketform('affine',[cosd(theta1) -sind(theta1) 0; sind(theta1) cosd(theta1) 0; 0 0 1]);
[imo, ptsy, ptsx] = transformImage(im, pts(:,2), pts(:,1), tform);

%zeros should remain zeros
ptsx = ptsx.*double(pts(:,1)>0 & pts(:,2)>0);
ptsy = ptsy.*double(pts(:,1)>0 & pts(:,2)>0);

%crop the image and change the origin of the points
x = 1 + round(rand(1) * (size(imo,2) - size(im,2)-1));
y = 1 + round(rand(1) * (size(imo,1) - size(im,1)-1));
imo = imo(y:y+size(im,1)-1,x:x+size(im,2)-1,:);

%GT points
ptsx=ptsx-x +1;
ptsy=ptsy-y +1;

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