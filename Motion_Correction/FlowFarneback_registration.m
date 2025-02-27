function Y = FlowFarneback_registration(Y, motionSource, FlowFarneback_opts)
% FLOWFARNEBACK_REGISTRATION - Corrects motion in `Y` using optical flow from `motionSource`
% This function estimates optical flow using Farneback's method on `motionSource`
% and applies the computed transformations to `Y`.
%
% INPUTS:
%   Y - 3D matrix (X, Y, T) representing the image stack to be corrected.
%   motionSource - 3D matrix used to estimate motion (e.g., a fluorescence channel).
%   FlowFarneback_opts - (Optional) Struct with Farneback optical flow parameters.
%
% OUTPUT:
%   Y - Motion-corrected image stack.

% ==============================  
% 1. Set Default Parameters  
% ==============================
if ~exist("FlowFarneback_opts", "var")
    % Default settings for Farneback Optical Flow
    FlowFarneback_opts.NumPyramidLevels = 3;   % Multi-scale pyramid levels
    FlowFarneback_opts.NeighborhoodSize = 7;   % Size of local neighborhood for motion estimation
    FlowFarneback_opts.FilterSize = 30;        % Gaussian filter size for motion vectors
    FlowFarneback_opts.NumIterations = 3;      % Number of refinement iterations
end

% ==============================  
% 2. Move Data to GPU (If Available)  
% ==============================
if canUseGPU
    motionSource = gpuarray(motionSource); % Move motion estimation source to GPU
end
corrected = motionSource; % Initialize corrected frames for tracking cumulative motion

% ==============================  
% 3. Initialize Optical Flow Object  
% ==============================
opticFlow = opticalFlowFarneback(FlowFarneback_opts); % Create optical flow object
reset(opticFlow) % Ensure a fresh start
ref=corrected(:,:,1);
% ==============================  
% 4. Compute and Apply Optical Flow for Motion Correction  
% ==============================
for t = progress(2:size(corrected, 3))
    % Step 1: Estimate optical flow between the last corrected frame and the new raw frame
    flow = estimateFlow(opticFlow, corrected(:,:,t-1)); % Use last corrected frame as reference
    flow = estimateFlow(opticFlow, motionSource(:,:,t)); % Compute flow from motionSource(t) to corrected(t-1)

    % Step 2: Construct the displacement field (flow vectors)
    D(:,:,:,t) = cat(3, flow.Vx, flow.Vy); % Store motion vector field

    % Step 3: Apply motion correction using the estimated displacement field
    corrected(:,:,t) = imwarp(motionSource(:,:,t), D(:,:,:,t)); % Warp motion estimation source
end

% ==============================  
% 5. Apply Gaussian Smoothing to Displacement Field  
% ==============================
% Smooths the motion field to reduce noise and improve registration quality
D(:,:,1,:) = imgaussfilt3(squeeze(D(:,:,1,:)), [0.5, 0.5, 0.5]); % Smooth X-component of flow
D(:,:,2,:) = imgaussfilt3(squeeze(D(:,:,2,:)), [0.5, 0.5, 0.5]); % Smooth Y-component of flow

% ==============================  
% 6. Warp Original Image Stack Using Corrected Motion Field  
% ==============================
for i = progress(1:size(D,4), 'Title', 'Applying shifts')
    Y(:,:,i) = imwarp(Y(:,:,i), D(:,:,:,i), 'FillValues', 0); % Apply computed shifts to Y
end

dummy=1