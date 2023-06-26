function [wholeImage] = Intensity_Image_Summation(array, frame_size_x)
%Calculation of summed intensity image
%   Detailed explanation goes here

wholeImage=zeros(frame_size_x);
for bin = 1:13
    
    %disp(bin)
    spectral_bins_per_pixel_mesh = array(:,bin,:);
    spectral_bins_per_pixel_mesh_reshaped = reshape(spectral_bins_per_pixel_mesh,[512 256 256]);
    
    for spectral_pixel = 1:400
        spectral_bins_per_pixel_mesh_reshapedoneFrame = reshape(spectral_bins_per_pixel_mesh_reshaped(spectral_pixel,:,:), [256 256]);
        wholeImage = wholeImage+spectral_bins_per_pixel_mesh_reshapedoneFrame;
        
    end
    
end


end

