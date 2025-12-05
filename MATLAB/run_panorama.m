clear all;
clc;
tic;
%buildingDir = fullfile(toolboxdir("vision"),"visiondata","building");
%buildingScene = imageDatastore(buildingDir);

%montage(buildingScene.Files)

%I = readimage(buildingScene,1);
% determine script folder (works on macOS, Linux, Windows)
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end

% images are located relative to the script (../images/mountains)
imagesDir = fullfile(scriptDir, '..', 'images', 'mountain');
refPath = fullfile(imagesDir, 'reference.jpg');
regPath = fullfile(imagesDir, 'registered.jpg');

I = imread(refPath);

grayImage = im2gray(I);
imshow(grayImage);
points = detectSURFFeatures(grayImage);
[features,points] = extractFeatures(grayImage,points);

images_list = [string(refPath), string(regPath)];
numImages = numel(images_list);

tforms(numImages) = projtform2d;
imageSize = zeros(numImages, 2);

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

% Blend non-reference images first, then place the reference on top
blendOrder = [2:numImages 1];
for ii = blendOrder
    I = imread(images_list(ii));
    warpedImage = imwarp(I,tforms(ii),OutputView=panoramaView);
    mask = imwarp(true(size(I,1),size(I,2)),tforms(ii),OutputView=panoramaView);
    panorama = imblend(warpedImage,panorama,mask,foregroundopacity=1);
end

imshow(panorama)
toc;
