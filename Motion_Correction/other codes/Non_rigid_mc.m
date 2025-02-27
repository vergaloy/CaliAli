function Y = Non_rigid_mc(Y, ref, opt)
%% Non_rigid_mc: Perform non-rigid motion correction using multi-level registration.
%
% This function applies non-rigid motion correction to an input video.
% The correction is performed using multi-level registration, incorporating
% optical flow for motion estimation.
%
% Inputs:
%   Y   - Input video (3D array)
%   ref - Reference image for alignment
%   opt - Options structure containing registration parameters
%
% Outputs:
%   Y   - Motion-corrected video

% Author: Pablo Vergara
% Contact: pablo.vergara.g@ug.uchile.cl
% Date: 2025

% Generate a pyramid of images (e.g., blood vessel and neuron projections)
fprintf('Applying non-rigid motion correction...\n');
[X] = get_video_pyramid(Y, ref, opt);

% Perform non-rigid motion correction in parallel
Y = NR_motion_correction_parallel(X, Y, opt);
end


function Y = NR_motion_correction_parallel(X, Y, opt)
% NR_motion_correction_parallel: Applies non-rigid motion correction in parallel.
%
% Inputs:
%   X   - 4D array of image pyramids for registration
%   Y   - Original video
%   opt - Options structure
%
% Outputs:
%   Y   - Motion-corrected video
for i = size(X,3):-1:1
    motionSource = squeeze(X(:,:,i,:));

    % Calculate motion scores for adaptive processing
    ms = get_motion_score(motionSource);

    % Distribute the video into smaller batches based on motion scores
    [MS, motionSource] = distribute(motionSource, ms, opt.non_rigid_batch_size);

    % Perform motion correction on each batch in parallel
    [Y, X] = MC_in(MS, motionSource,X,Y, opt.non_rigid_options);
end

end


function X = get_video_pyramid(Y, ref, opt)
% get_video_pyramid: Creates a multi-level image pyramid for registration.
%
% Inputs:
%   Y   - Input video (3D array)
%   ref - Reference image for alignment
%   opt - Structure containing registration options
%
% Outputs:
%   X   - 4D array of images (pyramid) for motion correction

order = opt.non_rigid_pyramid;  % Get the order of pyramid projections

% Apply Gaussian filtering based on selected reference projection type
switch opt.reference_projection_rigid
    case 'BV'
        BV = imgaussfilt3(ref, [2,2,2]);
    case 'centroid'
        Cen = ref;
    case 'high_pass'
        HP = ref;
end
clear ref;

% Compute image projections if they do not exist
if ismember('BV', order) && ~exist('BV','var')
    BV = CaliAli_get_blood_vessels(Y, opt);
    BV = imgaussfilt3(BV, [2,2,2]);
end
if ismember('neuron', order) && ~exist('Neu','var')
    opt.preprocessing.detrend=false;
    opt.preprocessing.noise_scale=false;
    Neu = v2uint8(CaliAli_remove_background(Y, opt));
end
if ismember('high_pass', order) && ~exist('HP','var')
    HP = get_filtered_projection(Y, opt);
end

% Generate centroid projection
if ismember('centroid', order)
    Neu = v2uint8(Neu);
    thr = prctile(Neu(Neu>0), 5);
    for i = progress(1:size(Neu,3), 'Title', 'Getting centroids')
        img = Neu(:,:,i) > thr;
        stats = regionprops(img, 'Centroid');
        dot_img = false(size(img));

        for k = 1:length(stats)
            centroid = round(stats(k).Centroid);
            dot_img(centroid(2), centroid(1)) = 1;
        end
        Cen(:,:,i) = adapthisteq(imgaussfilt(single(dot_img), 2), 'NumTiles', [4 4], 'ClipLimit', 0.8);
    end
end

% Construct the pyramid
X = [];
order = flip(order);
for i = 1:numel(order)
    switch order{i}
        case 'BV'
            X = cat(4, X, v2uint8(BV));
        case 'neuron'
            X = cat(4, X, Neu);
        case 'centroid'
            X = cat(4, X, v2uint8(Cen));
        case 'high_pass'
            X = cat(4, X, v2uint8(HP));
    end
end

% Reorder dimensions for processing
X = permute(X, [1, 2, 4, 3]);
end

function score = get_motion_score(motionSource)
% get_motion_score: Computes motion scores using optical flow.
%
% Inputs:
%   motionSource - 3D array representing motion images
%
% Outputs:
%   score - Computed motion score per frame

opticFlow = opticalFlowFarneback('NumPyramidLevels',1, 'NumIterations',2);
reset(opticFlow)
flow = estimateFlow(opticFlow, motionSource(:,:,1));

for i = 2:size(motionSource,3)
    flow = estimateFlow(opticFlow, motionSource(:,:,i-1));
    D(:,:,:,i) = cat(3, flow.Vx, flow.Vy);
end

% Apply Gaussian smoothing
D(:,:,1,:) = imgaussfilt3(squeeze(D(:,:,1,:)), [2,2,2]);
D(:,:,2,:) = imgaussfilt3(squeeze(D(:,:,2,:)), [2,2,2]);

% Compute mean absolute motion score
score = squeeze(mean(squeeze(abs(D(:,:,1,:)) + abs(D(:,:,2,:))), [1,2]));
end

function [MS, motionSource_out] = distribute(motionSource, ms, win)
% distribute: Divides video into smaller batches for parallel processing.
%
% Inputs:
%   motionSource - 4D array of images for registration
%   ms           - Motion scores for batch segmentation
%   win          - Batch size constraints [min, max]
%
% Outputs:
%   MS                - Cell array of motion scores per batch
%   motionSource_out  - Cell array of batch-segmented motion data

disp('Distributing video into small batches...');

v = movmedian(ms < prctile(ms, 50), 50) > 0.5;
v(1:win(2):end) = 0;  % Ensure minimum spacing
CC = bwlabel(v);

rp = regionprops(logical(CC), 'area');
for i = 1:max(CC)
    if rp(i).Area < win(1)
        CC(CC == i) = 0;
        CC(CC > i) = CC(CC > i) - 1;
    end
end

CC(CC == 0) = nan;
CC = fillmissing(CC, 'nearest');

for i = 1:max(CC)
    MS{i} = ms(CC == i);
    motionSource_out{i} = motionSource(:,:,CC == i);
end
end

function [Y,X] = MC_in(MS, MotionSource,X, Y, opt)
% Perform motion correction within each batch in parallel
% % 
[~,I]=cellfun(@(x) min(x),MS,'UniformOutput', false);

M=cellfun(@(x,y) x(:,:,y),MotionSource,I,'UniformOutput', false);
M=cat(3,M{:});

D=cell(1,length(MS));
parfor k = 1:length(MS)
     [D{k},Gout{k}] = batch_register(MotionSource{k},opt,MS{k},M(:,:,k));  % Register the batch
end

% Warp the video frames based on the calculated displacement fields

D=cat(4,D{:});

D(:,:,1,:)=imgaussfilt3(squeeze(D(:,:,1,:)),[0.5,0.5,0.5]);
D(:,:,2,:)=imgaussfilt3(squeeze(D(:,:,2,:)),[0.5,0.5,0.5]);

for i = progress(1:size(D,4),'Title','Applying shifts')
    Y(:,:,i) = imwarp(Y(:,:,i), D(:,:,:,i), 'FillValues', 0);
    for k = 1:size(X,3)
        X(:,:,k,i) = imwarp(squeeze(X(:,:,k,i)), D(:,:,:,i), 'FillValues', 0);
    end
end

end

function [D,G] = batch_register(motionSource,opt,ms,ref)
[~,ix]=min(ms);
if ~exist("ref","var")
ref=motionSource(:,:,ix);
end

pad_size=30;
motionSource = padarray(motionSource, [pad_size, pad_size, 0], "replicate", 'both');

[d1,d2,d3]=size(motionSource);
D=zeros(d1,d2,2,d3);

[~,I]=sort(abs((1:d3)-ix),'ascend');

ref = padarray(ref, [pad_size, pad_size, 0], "replicate", 'both');
opticFlow = opticalFlowFarneback(opt);
ref_temp=ref;
for i = ix:-1:1
    flow = estimateFlow(opticFlow, ref_temp);
    flow = estimateFlow(opticFlow, motionSource(:,:,i));
    D(:,:,:,i) = cat(3, flow.Vx, flow.Vy);
    D(pad_size+1:end-pad_size, pad_size+1:end-pad_size, :,:)=0;
    ref_temp = imwarp(motionSource(:,:,i), D(:,:,:,i)); 
    G(:,:,i)=ref_temp;
end
ref_temp=ref;

for i = ix+1:1:d3
    flow = estimateFlow(opticFlow, ref_temp);
    flow = estimateFlow(opticFlow, motionSource(:,:,i));
    D(:,:,:,i) = cat(3, flow.Vx, flow.Vy);
    ref_temp = imwarp(motionSource(:,:,i), D(:,:,:,i)); 
    G(:,:,i)=ref_temp;
end

D = D(pad_size+1:end-pad_size, pad_size+1:end-pad_size, :,:);

end
