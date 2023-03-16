function [wholeImage] = Intensity_Image_Summation(imageAlphaData)
%Calculation of summed intensity image
%   Detailed explanation goes here

imagedata = imageAlphaData(1:410,:,:);
wholeImage=squeeze(sum(imagedata,1));


end

