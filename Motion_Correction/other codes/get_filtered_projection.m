function Y=get_filtered_projection(Y,opt)

%% preprocessing data
% create a spatial filter for removing background
Y=single(Y);
psf = fspecial('gaussian', round(opt.gSig*4), round(opt.gSig*4));


Y = Y-imfilter(Y, psf, 'replicate');

Y=v2uint8(Y-median(Y,3));




