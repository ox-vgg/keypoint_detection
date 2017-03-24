function heatmap = generateGHeatmap(heatmap,cxy,theta,len,opts)

x1 = 1:1:size(heatmap,1); %rows
x2 = 1:1:size(heatmap,2); %cols

[X1,X2] = meshgrid(x1, x2);
sigma1 = len*opts.facX; %sigmx (part length)
sigma2 = len*opts.facY; %sigy
scale1 = 1;
scale2 = 1;
sigma1 = scale1*sigma1;
sigma2 = scale2*sigma2;
theta = 180-theta;

a = ((cosd(theta)^2) / (2*sigma1^2)) + ((sind(theta)^2) / (2*sigma2^2));
b = -((sind(2*theta)) / (4*sigma1^2)) + ((sind(2*theta)) / (4*sigma2^2));
c = ((sind(theta)^2) / (2*sigma1^2)) + ((cosd(theta)^2) / (2*sigma2^2));

mu(1)= cxy(1);
mu(2)= cxy(2);

%add up the heatmap for each individual
newMap = exp(-(a*(X1 - mu(1)).^2 + 2*b*(X1 - mu(1)).*(X2 - mu(2)) + c*(X2 - mu(2)).^2));

newMap(isnan(newMap))=0;
newMap(newMap<10^-18)=0; %cut the tails

heatmap = heatmap + (opts.magnif*newMap);