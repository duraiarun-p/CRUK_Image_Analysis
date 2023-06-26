function [wavelengths,wavenumbers] = Wavelength_Calculator()
%Calculate wavelength and wavenumbers for the sensor pixels


excitation_wavelength = 485;
%slope_wavelength_vs_pixel = 0.729167;
%C = 520;
slope_wavelength_vs_pixel = 0.5468;
C = 460;
pixels = 1:1:512;
%wavelengths = zeros(512);
wavelengths = C + slope_wavelength_vs_pixel*pixels;


for i = 1:1:512
    wavelength_val = C + slope_wavelength_vs_pixel*i;
    wavenumbers(i) = (wavelength_val - excitation_wavelength)*10^7/(wavelength_val*excitation_wavelength);
end


end

