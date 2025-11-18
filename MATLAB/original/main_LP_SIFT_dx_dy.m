% Clear the command window
clear;
% Close all open figure windows
close all;
% Start the stopwatch timer
tic;

% Define the path where the data images are stored
pathname = ('.\data\');
% Define the filenames of the original images
ori_img1_filename = ('data1_ori_top_left_corner.tif');
ori_img2_filename = ('data1_ori_bottom_right_corner.tif');

% Read the original images from the specified path
ori_img1 = imread(strcat(pathname,ori_img1_filename));
ori_img2 = imread(strcat(pathname,ori_img2_filename));

% Add line noise to the grayscale version of the original images to break the extreme value lock
% Convert the original images to grayscale and then to double precision
distimg1 = add_line_noise(double(img2gray(ori_img1)));
distimg2 = add_line_noise(double(img2gray(ori_img2)));

% Display the noisy images side by side
subplot(1,2,1);imshow(uint8(distimg1));
subplot(1,2,2);imshow(uint8(distimg2));

% Define a parameter q used for feature point detection
q = [256,512]; 
% Detect feature points in the original and noisy images using the Test_feature_points_DE function
kpts1 = Test_feature_points_DE(ori_img1,distimg1,q);
kpts2 = Test_feature_points_DE(ori_img2,distimg2,q);
% Compute descriptors for the detected feature points in the original images
descriptors1 = Compute_descriptors(ori_img1,kpts1);
descriptors2 = Compute_descriptors(ori_img2,kpts2);

% Display the noisy image 1 and mark the detected feature points
subplot(1,2,1);imshow(uint8(distimg1));
hold on
% Set the background color of the current figure to white
set(gcf,'Color','w');
% Plot the detected feature points as red stars
plot(kpts1(:,2),kpts1(:,1),'r*', 'Linewidth', 4,'MarkerSize', 4);
% Set the title of the subplot
title('test feature points');
hold off

% Display the noisy image 2 and mark the detected feature points
subplot(1,2,2);imshow(uint8(distimg2));
hold on
% Set the background color of the current figure to white
set(gcf,'Color','w');
% Plot the detected feature points as red stars
plot(kpts2(:,2),kpts2(:,1),'r*','Linewidth', 4,'MarkerSize', 4);
% Set the title of the subplot
title('test feature points');
hold off

% Define a threshold for feature matching
threshold = 0.01;  

% Match the feature points between the two images and get the matched pairs and their locations
[matched, locs1, locs2] = drawMatched(kpts1, kpts2, descriptors1, descriptors2, threshold);
% Stitch the two original images together using the matched feature points
[stitched_img,H] = stitch(ori_img1, ori_img2, locs1, locs2, 3);
% Create a new figure to display the stitched image
figure;
% Display the stitched image
imshow(uint8(stitched_img));
% Set the title of the figure
title('stitching result');

% Save the stitched image to the specified path with a generated filename
filename = strcat(pathname,'LP-SIFT result',{' '},ori_img1_filename(1:end-4),' and',{' '},ori_img2_filename);
% Convert the stitched image to uint8 data type
temp_uint8 = uint8(stitched_img);
% Write the stitched image to a file
imwrite(temp_uint8,filename{1,1});
% Stop the stopwatch timer and display the elapsed time
toc;