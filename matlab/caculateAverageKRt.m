close all; clear all;
% read in camera parameters
load('./stereoParams.mat');
cameraParams_1 = stereoParams.CameraParameters1;
cameraParams_2 = stereoParams.CameraParameters2;
numPatterns = stereoParams.NumPatterns;

% intialise arrays used in calculation
sum = zeros(3, 3);
sum_KRt = zeros(3,3);
Rts = zeros(numPatterns, 1);
KRts = zeros(numPatterns, 1);

for i = 1:numPatterns
    % RGB camera
    extrinsics_1 = cameraParams_1.PatternExtrinsics(i);
    intrinsics_1 = cameraParams_1.Intrinsics;
    % Event Camera
    extrinsics_2 = cameraParams_2.PatternExtrinsics(i);
    intrinsics_2 = cameraParams_2.Intrinsics;

    % calculate K*[R t] for each camera
    KRt_1 = intrinsics_1.K*[extrinsics_1.R extrinsics_1.Translation']; % K[R t]
    KRt_2 = intrinsics_2.K*[extrinsics_2.R extrinsics_2.Translation']; % K[R t]

    % calculate [R t] for each camera
    Rt_1 = [extrinsics_1.R extrinsics_1.Translation'];
    Rt_2 = [extrinsics_2.R extrinsics_2.Translation'];

    % divide extrinsics
    divRt = Rt_1/Rt_2;
    % increment sum used for average
    sum = sum + divRt;
    
    %divide KRt
    divKRt = KRt_1/KRt_2;
    % increment sum used for average
    sum_KRt=sum_KRt+divKRt;

end
av_divRt = sum./numPatterns; % average extrinsics
av_divKRt = sum_KRt./numPatterns; %  K1/K2*average extrinsics
