clear all;
clc;
tic;
%buildingDir = fullfile(toolboxdir("vision"),"visiondata","building");
%buildingScene = imageDatastore(buildingDir);

%montage(buildingScene.Files)

%I = readimage(buildingScene,1);
I = imread("C:\repos\CSS587\css587project\images\mountains\reference.jpg");

grayImage = im2gray(I);
imshow(grayImage);
points = detectSURFFeatures(grayImage);
[features,points] = extractFeatures(grayImage,points);

%numImages = numel(buildingScene.Files);
numImages = 2;
tforms(numImages) = projtform2d;

imageSize = zeros(numImages, 2);

images_list = [
    "C:\repos\CSS587\css587project\images\mountains\reference.jpg",
    "C:\repos\CSS587\css587project\images\mountains\registered.jpg"];

for n = 2:numImages
    pointsPrevious = points;
    featuresPrevious = features;

    %I = readimage(buildingScene, n);
    I = imread(images_list(n));
    imshow(I);
    grayImage = im2gray(I);
    imageSize(n,:) = size(grayImage);

    points = detectSURFFeatures(grayImage);
    [features,points] = extractFeatures(grayImage,points);

    indexPairs = matchFeatures(features,featuresPrevious,Unique=true);
    matchedPoints = points(indexPairs(:,1),:);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);

    tforms(n) = estgeotform2d(matchedPoints, matchedPointsPrev,...
        "projective",Confidence=99.9,MaxNumTrials=2000);
    
    tforms(n).A = tforms(n-1).A * tforms(n).A;
end

for idx = 1:numel(tforms)
    [xlim(idx,:),ylim(idx,:)] = outputLimits(tforms(idx),[1 imageSize(idx,2)],[1 imageSize(idx,1)]);
end

avgLim = mean(xlim, 2);
[~,idx] = sort(avgLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);

Tinv = invert(tforms(centerImageIdx));
for idx = 1:numel(tforms)
    tforms(idx).A = Tinv.A * tforms(idx).A;
end

for idx = 1:numel(tforms)
    [xlim(idx,:),ylim(idx,:)] = outputLimits(tforms(idx),[1 imageSize(idx,2)],[1 imageSize(idx,1)]);
end
maxImageSize = max(imageSize);

xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);
yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

width = round(xMax - xMin);
height = round(yMax - yMin);

panorama = zeros([height width 3],"like",I);

xLimits = [xMin, xMax];
yLimits = [yMin, yMax];

panoramaView = imref2d([height width], xLimits, yLimits);

for idx = 1:numImages
    %I = readimage(buildingScene,idx);
    I = imread(images_list(idx));
    warpedImage = imwarp(I,tforms(idx),OutputView=panoramaView);
    mask = imwarp(true(size(I,1),size(I,2)),tforms(idx),OutputView=panoramaView);
    panorama = imblend(warpedImage,panorama,mask,foregroundopacity=1);
end

imshow(panorama)
toc;