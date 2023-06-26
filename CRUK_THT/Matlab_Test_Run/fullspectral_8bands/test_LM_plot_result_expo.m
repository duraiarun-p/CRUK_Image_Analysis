
%% plot image
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
% ylim([0 5]);
% legend

% subplot(2,2,1);
% selected_old_cpu = reshape(parameters_old_cpu(2,:), 512, 256, 256);
% selected_old_cpu = reshape(selected_old_cpu(spectral_pixel, :,:), 256, 256);
% imagesc(selected_old_cpu);
% colorbar;
% title("Old cpu");
% caxis([0 5]);
% 
% subplot(2,2,2);
% selected_old_gpu = reshape(parameters_old_gpu(2,:), 512, 256, 256);
% selected_old_gpu = reshape(selected_old_gpu(spectral_pixel, :,:), 256, 256);
% imagesc(selected_old_gpu);
% colorbar;
% title("Old gpu");
% caxis([0 5]);

spectral_pixel = 50;

%clim = [0 5];
%subplot(1,3,1);
selected_cpu = reshape(parameters_cpu(2,:),512, 256, 256);
selected_cpu = reshape(selected_cpu(spectral_pixel, :,:), 256, 256);
imagesc(selected_cpu);
colorbar;
title("Cpufit");
caxis([0 5]);

% subplot(1,3,2);
% selected_gpu = reshape(parameters_gpu(2,:), 256, 256);
% imagesc(selected_gpu);
% colorbar;
% title("Gpufit");
% caxis(clim);
% 
% subplot(1,3,3);
% selected_gpu = reshape(parameters_ahmet(2,:), 256, 256);
% imagesc(selected_gpu);
% colorbar;
% title("Ahmet");
% caxis(clim);

%% plot fitting
selected_pixel = 1;
exp_fitting = @(x, params) params(1)*exp(-x/params(2)) + params(3);
binWidth = 0.8;
selected_bins = (3:14)';
x_points = linspace(0, size(selected_bins, 1) - 1, size(selected_bins, 1))'*binWidth;

subplot(1,3,1);
plot(x_points, selected_data_for_fitting(:, selected_pixel), "bo");
hold on;
%plot(x_points, exp_fitting(x_points, parameters_ahmet(:, selected_pixel)));
plot(x_points, exp_fitting(x_points, parameters_cpu(:, selected_pixel)));
%plot(x_points, exp_fitting(x_points, parameters_gpu(:, selected_pixel)));
legend("Original points","old exponential fitting", "Cpufit fitting", "Gpufit fitting");

subplot(1,3,2);
selected_pixel = 3;
plot(x_points, selected_data_for_fitting(:, selected_pixel), "bo");
hold on;
plot(x_points, exp_fitting(x_points, parameters_ahmet(:, selected_pixel)));
plot(x_points, exp_fitting(x_points, parameters_cpu(:, selected_pixel)));
plot(x_points, exp_fitting(x_points, parameters_gpu(:, selected_pixel)));
legend("Original points","old exponential fitting", "Cpufit fitting", "Gpufit fitting");

subplot(1,3,3);159159

selected_pixel = 4;
plot(x_points, selected_data_for_fitting(:, selected_pixel), "bo");
hold on;
plot(x_points, exp_fitting(x_points, parameters_ahmet(:, selected_pixel)));
plot(x_points, exp_fitting(x_points, parameters_cpu(:, selected_pixel)));
plot(x_points, exp_fitting(x_points, parameters_gpu(:, selected_pixel)));
legend("Original points","old exponential fitting", "Cpufit fitting", "Gpufit fitting");

