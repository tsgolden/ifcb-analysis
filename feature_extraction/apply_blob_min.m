function [ target ] = apply_blob_min( target )
%function [ img_blob ] = apply_blob_min( img_blob )
% take an input b&w blob image, remove any continguous components <
% blob_min, and return the resulting blob image along with the summed area
% of the remaining components 

blob_min = target.config.blob_min;
img_cc = bwconncomp(target.blob_image);
t = regionprops(img_cc, 'Area');
idx = find([t.Area] > blob_min);
target.blob_image = ismember(labelmatrix(img_cc), idx); %is this most efficient method?
%target.blob_props.Area = [t(idx).Area];
target.blob_props.Area = t(idx).Area;
end

