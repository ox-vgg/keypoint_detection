function [new_im, grn, gcn] = transformImage(I, gr, gc, TM)
%More info at: http://stackoverflow.com/questions/13366771/matlab-image-transformation

[new_im, xdata, ydata] = imtransform(I, TM,'XYScale',1,'FillValues', 128);
w = xdata(2)-xdata(1) +1;
h = ydata(2)-ydata(1)+1;
scalex = size(new_im,2)/w;
scaley = size(new_im,1)/h;

coords = [gc(:), gr(:)];
coords_tf = tformfwd(TM, coords);

%translation
coords_tf_mg(:,1) = coords_tf(:,1) - xdata(1) + 1;
coords_tf_mg(:,2) = coords_tf(:,2) - ydata(1) + 1;

%scale
coords_tf_mg(:,1) = coords_tf_mg(:,1)*scalex;
coords_tf_mg(:,2) = coords_tf_mg(:,2)*scaley;

coords_tf_mg = round(coords_tf_mg);
grn = coords_tf_mg(:,2);
gcn = coords_tf_mg(:,1);
grn = reshape(grn, size(gr,1), size(gr,2));
gcn = reshape(gcn, size(gc,1), size(gc,2));
end