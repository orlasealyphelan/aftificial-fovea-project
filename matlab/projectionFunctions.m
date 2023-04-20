close all; clear all;
load('stereoParams.mat');
cameraParams_1 = stereoParams.CameraParameters1;
cameraParams_2 = stereoParams.CameraParameters2;
RGBimages = dir('./8.9/*.bmp');
EDimages = dir('./ED/*.bmp');
nfiles = length(RGBimages);
plot_camera = 'RGB'; % change to 'RGB' to plot RGB projections

for i = 1:nfiles
    % read in images
    name1 = RGBimages(i).name;
    name2 = EDimages(i).name;
    I1 = imread(strcat(RGBimages(i).folder,'/',name1));
    I2 = imread(strcat(strcat(EDimages(i).folder,'/',name2)));
    
    % undistort images
    I1 = undistortImage(I1, stereoParams.CameraParameters1.Intrinsics);
    I2 = undistortImage(I2, stereoParams.CameraParameters2.Intrinsics);
    
    % detect checkerboard points in each image
    imagePoints_1 = detectCheckerboardPoints(I1);
    imagePoints_2 = detectCheckerboardPoints(I2);
    
     % convert detected checkerboard points to wold points for each camera
     worldPoints_1 = img2world2d(imagePoints_1, cameraParams_1.PatternExtrinsics(i), cameraParams_1.Intrinsics);
     worldPoints_2 = img2world2d(imagePoints_2, cameraParams_2.PatternExtrinsics(i), cameraParams_2.Intrinsics);
     zCoord_1 = zeros(size(worldPoints_1,1),1);
     zCoord_2 = zeros(size(worldPoints_2,1),1);
     worldPoints_1 = [worldPoints_1 zCoord_1];
     worldPoints_2 = [worldPoints_2 zCoord_2];
    
     % project event world points to image points for the 8.9 camera
     projectedPoints_2 = world2img(worldPoints_2, cameraParams_1.PatternExtrinsics(i), cameraParams_1.Intrinsics);
     % project RGB world points to image points for the ED camera
     projectedPoints_1_ED = world2img(worldPoints_1, cameraParams_2.PatternExtrinsics(i), cameraParams_2.Intrinsics);

     if strcmp(plot_camera, 'RGB')
         % plot both sets of projected points onto 8.9 camera image
         fig1 = figure('Name', name1);
         imshow(I1);
         hold on
         plot(imagePoints_1(:,1),imagePoints_1(:,2),"b*-");
         plot(projectedPoints_2(:,1),projectedPoints_2(:,2),"r*-");
         legend("Detected Corners", "Projection from ED");
         hold off
     end
    
     if strcmp(plot_camera, 'ED')
         % plot both sets of projected points onto ED camera image
         fig2 = figure('Name', name2);
         imshow(I2);
         hold on
         plot(imagePoints_2(:,1),imagePoints_2(:,2),"b*-");
         plot(projectedPoints_1_ED(:,1),projectedPoints_1_ED(:,2),"r*-");
         legend("Detected Corners", "Projection from RGB");
         hold off
     end
end
