spectral_pixel = 50;

% subplot(3,2,[1 2]);
% mean_old_cpu = reshape(parameters_old_cpu(2,:), 512, 256*256);
% mean_old_cpu = mean(mean_old_cpu, 2);
% plot(mean_old_cpu);
% hold on;
% mean_old_gpu = reshape(parameters_old_gpu, 512, 256*256);
% mean_old_gpu = mean(mean_old_gpu, 2);
% plot(mean_old_gpu);
% mean_cpu = reshape(parameters_cpu(2,:), 512, 256*256);
% mean_cpu = mean(mean_cpu, 2);
% plot(mean_cpu);
% mean_gpu = reshape(parameters_gpu(2,:), 512, 256*256);
% mean_gpu = mean(mean_gpu, 2);
% plot(mean_gpu);
% % ylim([0 5]);
% % legend
% 
% subplot(2,2,1);
% selected_old_cpu = reshape(parameters_old_cpu(2,:), 512, 256, 256);
% selected_old_cpu = reshape(selected_old_cpu(spectral_pixel, :,:), 256, 256);
% imagesc(selected_old_cpu);
% colorbar;
% title("Old cpu");
% caxis([0 5]);
% 
% subplot(2,2,2);
% selected_old_gpu = reshape(parameters_old_gpu, 512, 256, 256);
% selected_old_gpu = reshape(selected_old_gpu(spectral_pixel, :,:), 256, 256);
% imagesc(selected_old_gpu);
% colorbar;
% title("Old gpu");
% caxis([0 5]);
%%
wholeImage=zeros(256);
for bin = 1:16
    
    disp(bin)
    spectral_bins_per_pixel_mesh = bins_array_3(:,bin,:);
    spectral_bins_per_pixel_mesh_reshaped = reshape(spectral_bins_per_pixel_mesh,[512 256 256]);
    
    for spectral_pixel = 1:400
        spectral_bins_per_pixel_mesh_reshapedoneFrame = reshape(spectral_bins_per_pixel_mesh_reshaped(spectral_pixel,:,:), [256 256]);
        wholeImage = wholeImage+spectral_bins_per_pixel_mesh_reshapedoneFrame;
        
    end
    
end
%%
spectral_pixel = 50;
BackgroundImage = wholeImage;
BackgroundImage(BackgroundImage<40000) =40000;
BackgroundImage=BackgroundImage-40000;

mask = (BackgroundImage>5);

subplot(2,2,1);
imagesc(BackgroundImage);

subplot(2,2,2);
imagesc(mask);

selected_cpu = reshape(parameters_cpu(2,:), 512, 256, 256);
selected_cpu = reshape(selected_cpu(spectral_pixel, :,:), 256, 256);
selected_cpu(selected_cpu>10) = 0;
selected_cpu(selected_cpu<0) = 0;

subplot(2,2,3);
selcted_cpu_masked = selected_cpu.*mask;
imagesc(selcted_cpu_masked);
colorbar;
title("Cpufit");
%caxis([0 6]);

% subplot(2,2,4);
% selected_gpu = reshape(parameters_gpu(2,:), 512, 256, 256);
% selected_gpu = reshape(selected_gpu(spectral_pixel, :,:), 256, 256);
% imagesc(selected_gpu);
% colorbar;
% title("Gpufit");
% caxis([0 5]);