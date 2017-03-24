% --------------------------------------------------------------------
function imdb = getImdbNoAug(opts)
% --------------------------------------------------------------------

% Load the data to form the imdb file

load(opts.DataMatTrain); %training data

imdb.images.data=imgPath;
sets=ones(1,numel(imgPath));
imdb.images.labels=ptsAll;

clear imgPath ptsAll;

load(opts.DataMatVal); %validation data

sets=[sets 2*ones(1,numel(imgPath))];
imdb.images.data=[imdb.images.data imgPath];

if iscell(imdb.images.labels)%different formats of ground-truth
    imdb.images.labels=[imdb.images.labels ptsAll];
else
    imdb.images.labels=cat(3,imdb.images.labels,ptsAll);
end

imdb.images.set=sets;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.patchHei=opts.patchHei;
imdb.patchWi=opts.patchWi;
imdb.averageImage = [];