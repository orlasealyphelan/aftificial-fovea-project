close all; clear all;
load('stereoParams.mat');
cameraParams_1 = stereoParams.CameraParameters1;
cameraParams_2 = stereoParams.CameraParameters2;
RGBimages = dir('./8.9/*.bmp');
EDimages = dir('./ED/*.bmp');
nfiles = length(RGBimages);
plot_camera = 'ED'; % change to 'RGB' to plot RGB projections

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
    l1 = length(imagePoints_1);
    imagePoints_2 = detectCheckerboardPoints(I2);
    l2 = length(imagePoints_2);

    % RGB Camera
    extrinsics_1 = cameraParams_1.PatternExtrinsics(i);
    intrinsics_1 = cameraParams_1.Intrinsics;

    % Event Camera
    extrinsics_2 = cameraParams_2.PatternExtrinsics(i);
    intrinsics_2 = cameraParams_2.Intrinsics;

    % convert detected checkerboard points to wold points for each camera
    worldPoints_1 = img2world2d(imagePoints_1, extrinsics_1, intrinsics_1);
    worldPoints_2 = img2world2d(imagePoints_2, extrinsics_2, intrinsics_2);

    % calculate image points from world points using formula for 8.9 camera
    KRt = intrinsics_1.K*[extrinsics_1.R extrinsics_1.Translation']; % K[R t]
    
    XYZ1 = zeros(4,l1);
    for j = 1:l1
        XYZ1(:,j) = [worldPoints_1(j,1); worldPoints_1(j,2); 0; 1;]; %[X Y Z 1]'
    end
    imagePoints_calc = KRt*XYZ1/1000;

    % calculate image points from world points using formula for ED camera
    KRt_ED = intrinsics_2.K*[extrinsics_2.R extrinsics_2.Translation']; % K[R t]
    
    XYZ1_ED = zeros(4,l2);
    for j = 1:l2
        XYZ1_ED(:,j) = [worldPoints_2(j,1); worldPoints_2(j,2); 0; 1;]; %[X Y Z 1]'
    end
    imagePoints_calc_ED = KRt_ED*XYZ1_ED/1000;

    % calculate 8.9 image points from ED image points
    divKRt = KRt/KRt_ED;
    xy1 = zeros(3, l2);
    for k=1:l2
        xy1(:,k) = [imagePoints_2(k,1); imagePoints_2(k,2); 1];
    end

    imagPoints1_calc = divKRt*xy1;
    imagePoints1_calc_norm = zeros(2,length(imagPoints1_calc));
    for n=1:length(imagPoints1_calc)
        imagePoints1_calc_norm(1,n) = imagPoints1_calc(1,n)/imagPoints1_calc(3,n);
        imagePoints1_calc_norm(2,n) = imagPoints1_calc(2,n)/imagPoints1_calc(3,n);
    end

    if strcmp(plot_camera, 'RGB')
        % plot projected points onto 8.9 camera image
        figure('Name', name1)
        imshow(I1);
        hold on
        plot(imagePoints_1(:,1),imagePoints_1(:,2),"b*-");
        plot(imagePoints1_calc_norm(1,:),imagePoints1_calc_norm(2,:),"r*-");
        legend("Detected Points", "Projected Points from ED");
        hold off
    end

    % calculate ED image points from 8.9 image points
    divKRt_ED = KRt_ED/KRt;
    xy1_ED = zeros(3, l1);
    for k=1:l1
        xy1_ED(:,k) = [imagePoints_1(k,1); imagePoints_1(k,2); 1];
    end

    imagPoints2_calc = divKRt_ED*xy1_ED;
    imagPoints2_calc_norm = zeros(2,length(imagPoints2_calc));
    for n=1:length(imagPoints2_calc)
        imagPoints2_calc_norm(1,n) = imagPoints2_calc(1,n)/imagPoints2_calc(3,n);
        imagPoints2_calc_norm(2,n) = imagPoints2_calc(2,n)/imagPoints2_calc(3,n);
    end
    
    if strcmp(plot_camera, 'ED')
        % plot projected points onto ED camera image
        figure('Name', name2)
        imshow(I2);
        hold on
        plot(imagePoints_2(:,1),imagePoints_2(:,2),"b*-");
        plot(imagPoints2_calc_norm(1,:),imagPoints2_calc_norm(2,:),"r*-");
        legend("Detected Points", "Projected Points from 8.9");
        hold off
    end
end