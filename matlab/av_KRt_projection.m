close all; clear all;
% read in camera parameters and image directories
load('stereoParams.mat');
load('av_divKRt.mat');
RGBimages = dir('./8.9/*.bmp');
EDimages = dir('./ED/*.bmp');
nfiles = length(RGBimages);

% array to hold reprojection errors
errors = zeros(1, nfiles);

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
    l = length(imagePoints_2);

    % calculate 8.9 image points from ED image points
    xy1 = zeros(3, l);
    % add z-dimension
    for k=1:l
        xy1(:,k) = [imagePoints_2(k,1); imagePoints_2(k,2); 1];
    end

    imagPoints1_calc = zeros(3, l);
    % multiply by intrinsic and extrinsic matrices
    for j=1:l
        imagPoints1_calc(:, j) = av_divKRt*xy1(:, j);
    end

    % divide by z value to normalise
    imagePoints1_calc_norm = zeros(2,l);
    for n=1:l
        imagePoints1_calc_norm(1,n) = imagPoints1_calc(1,n)/imagPoints1_calc(3,n);
        imagePoints1_calc_norm(2,n) = imagPoints1_calc(2,n)/imagPoints1_calc(3,n);
    end
       
    % used to remove invalid points detected by detectCheckerBoardPoints
    if i ==18
        imagePoints_1 = imagePoints_1(1:63, :);
        remove = sort([8,22,29,36,50,57,43,15,1], 'descend');
        for j=1:length(remove)
            imagePoints_1(remove(j), :) = [];
        end
        hold off
    elseif i == 19
        imagePoints_1 = imagePoints_1(1:48, :);
    end
    calc_transpose = imagePoints1_calc_norm.';
    if i == 18
        imagePoints_1 = flip(imagePoints_1);
        for k=1:6:54
            imagePoints_1(k:k+5, :) = flip(imagePoints_1(k:k+5, :));
        end
    end
    if size(calc_transpose, 1) < size(imagePoints_1, 1)
        imagePoints_1 = imagePoints_1(7:54, :);
    end

    errors(i) = calculateError(calc_transpose, imagePoints_1);

    % plot both sets of projected points onto 8.9 camera image
    fig = figure('Name', name1);
    imshow(I1);
    hold on
    plot(imagePoints_1(:,1),imagePoints_1(:,2),"b*-");
    plot(calc_transpose(:,1),calc_transpose(:,2),"g*-");
    legend("Detected Points", "Projection from ED");
    hold off
    % exportgraphics(fig, 'highres_projection.png','Resolution',600)
    % saveas(fig, strcat('projection', name1))
end