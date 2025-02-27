function Y = Non_rigid_FlowFarneback(Y, motionSource, opt)
%% Non_rigid_mc: Perform non-rigid motion correction using multi-level registration.
 % Y = Non_rigid_FlowFarneback(Y,BV, CaliAli_options.motion_correction);
% Author: Pablo Vergara
% Contact: pablo.vergara.g@ug.uchile.cl
% Date: 2025

% Get a pyramid of images (e.g., blood vessel and neuron projections)
fprintf('Starting non-rigid motion correction...\n');

% Perform non-rigid motion correction in parallel
Y = NR_motion_correction_parallel(Y, v2uint8(motionSource), opt);
end


function Y = NR_motion_correction_parallel(Y,motionSource, opt)
% NR_motion_correction_parallel performs parallel non-rigid motion correction.
for i=1:2
% Calculate motion scores
ms = get_motion_score(motionSource);

% Distribute the video into smaller batches for parallel processing
[MS, motionSource] = distribute(motionSource,ms, opt.non_rigid_batch_size);

% Default settings for Farneback Optical Flow
FlowFarneback_opts.NumPyramidLevels = 3;   % Multi-scale pyramid levels
FlowFarneback_opts.NeighborhoodSize = 7;   % Size of local neighborhood for motion estimation
FlowFarneback_opts.FilterSize = 30;        % Gaussian filter size for motion vectors
FlowFarneback_opts.NumIterations = 3;      % Number of refinement iterations

% Perform motion correction on each batch in parallel
[Y,motionSource] = MC_in(MS, motionSource, Y,  FlowFarneback_opts);
end

end


function [MS, motionSource_out] = distribute(motionSource,ms, win)
% distribute divides a video into smaller batches for parallel processing.
%
%   [MS, G, Vc] = distribute(X, V, ms, win)
%
%   This function divides a video into smaller batches based on motion scores
%   and a specified window size.
%
%   Inputs:
%       X   - 4D array of images (pyramid) used for registration.
%       V   - Input video as a 3D array.
%       ms  - Motion scores used to determine batch boundaries.
%       win - 2-element vector specifying the minimum and maximum batch size.
%
%   Outputs:
%       MS  - Cell array of motion scores for each batch.
%       G   - Cell array of image pyramids for each batch.
%       Vc  - Cell array of video batches.

disp('Distributing video in small batches...');

% Identify regions of low motion to define batch boundaries
v = movmedian(ms < prctile(ms, 50), 50) > 0.5;
v(1:win(2):end) = 0;  % Ensure minimum spacing between batches
CC = bwlabel(v);  % Label connected components

% Remove small regions to ensure minimum batch size
rp = regionprops(logical(CC), 'area');
for i = 1:max(CC)
    if rp(i).Area < win(1)
        CC(CC == i) = 0;
        CC(CC > i) = CC(CC > i) - 1;
    end
end

% Fill gaps between batches using nearest neighbor interpolation
CC(CC == 0) = nan;
CC = fillmissing(CC, 'nearest');

% Distribute the video into batches
for i = 1:max(CC)
    MS{i} = ms(CC == i);
    motionSource_out{i} = motionSource(:,:,CC == i);
end
end

function [V,Gout] = MC_in(MS, G, V, opt)
% Perform motion correction within each batch in parallel
% % 
[~,I]=cellfun(@(x) min(x),MS,'UniformOutput', false);

M=cellfun(@(x,y) x(:,:,y),G,I,'UniformOutput', false);
M=cat(3,M{:});

% D = batch_register(M,opt,round(size(M,3)/2)); 
% for i = progress(1:size(D,4),'Title','Applying shifts')
%    M(:,:,i) = imwarp(M(:,:,i), D(:,:,:,i), 'FillValues', 0);
% end


D=cell(1,length(MS));
parfor k = 1:length(MS)
     [D{k},Gout{k}] = batch_register(G{k},opt,MS{k},M(:,:,k));  % Register the batch
end

% Warp the video frames based on the calculated displacement fields

D=cat(4,D{:});

D(:,:,1,:)=imgaussfilt3(squeeze(D(:,:,1,:)),[0.5,0.5,0.5]);
D(:,:,2,:)=imgaussfilt3(squeeze(D(:,:,2,:)),[0.5,0.5,0.5]);


for i = progress(1:size(D,4),'Title','Applying shifts')
    V(:,:,i) = imwarp(V(:,:,i), D(:,:,:,i), 'FillValues', 0);
end
Gout=cat(3,Gout{:});

end

function [D,G] = batch_register(motionSource,opt,ms,ref)
[~,ix]=min(ms);
if ~exist("ref","var")
ref=motionSource(:,:,ix);
end
[d1,d2,d3]=size(motionSource);
D=zeros(d1,d2,2,d3);

[~,I]=sort(abs((1:d3)-ix),'ascend');
ref_temp=ref;
opticFlow = opticalFlowFarneback(opt);
for i = ix:-1:1
    flow = estimateFlow(opticFlow, ref_temp);
    flow = estimateFlow(opticFlow, motionSource(:,:,i));
    D(:,:,:,i) = cat(3, flow.Vx, flow.Vy);

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

end

function score=get_motion_score(motionSource)
opticFlow = opticalFlowFarneback('NumPyramidLevels',1,'NumIterations',2); % Create optical flow object
reset(opticFlow)
flow = estimateFlow(opticFlow, motionSource(:,:,1));
for i=2:size(motionSource,3)
    flow = estimateFlow(opticFlow, motionSource(:,:,i-1));
    D(:,:,:,i) = cat(3, flow.Vx, flow.Vy);
end

D(:,:,1,:) = imgaussfilt3(squeeze(D(:,:,1,:)), [2, 2, 2]); % Smooth X-component of flow
D(:,:,2,:) = imgaussfilt3(squeeze(D(:,:,2,:)), [2, 2, 2]);
score=squeeze(mean(squeeze(abs(D(:,:,1,:))+abs(D(:,:,2,:))),[1,2]));
end
