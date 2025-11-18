% This function implements the Random Sample Consensus (RANSAC) algorithm to estimate a model (e.g., homography, fundamental matrix)
% from a set of point correspondences while robustly handling outliers.
% Input:
%   - x: A matrix representing the points in the first set. Each column corresponds to a point [x1; y1].
%   - y: A matrix representing the corresponding points in the second set. Each column corresponds to a point [x2; y2].
%   - ransacCoef: A structure containing the RANSAC algorithm parameters, including:
%       - minPtNum: Minimum number of points required to estimate the model.
%       - iterNum: Number of iterations for the RANSAC algorithm.
%       - thInlrRatio: Threshold for the inlier ratio.
%       - thDist: Distance threshold to determine if a point is an inlier.
%   - funcFindF: A function handle to estimate the model from a given set of point correspondences.
%   - funcDist: A function handle to calculate the distance between the estimated model and the points.
% Output:
%   - f: The estimated model (e.g., homography matrix) with the maximum number of inliers.
%   - inlierIdx: Indices of the inlier points in the original point sets.
function [f, inlierIdx] = RANSAC(x, y, ransacCoef, funcFindF, funcDist)
    % Extract the minimum number of points required to estimate the model.
    minPtNum = ransacCoef.minPtNum;
    % Extract the number of iterations for the RANSAC algorithm.
    iterNum = ransacCoef.iterNum;
    % Extract the threshold for the inlier ratio.
    thInlrRatio = ransacCoef.thInlrRatio;
    % Extract the distance threshold to determine inliers.
    thDist = ransacCoef.thDist;
    % Get the total number of points in the point sets.
    ptNum = size(x, 2);
    % Calculate the minimum number of inliers required based on the inlier ratio threshold.
    Inlier_thres = round(thInlrRatio * ptNum);
    % Initialize an array to store the number of inliers for each iteration.
    inlier_Num = zeros(1,iterNum);
    % Initialize a cell array to store the estimated models for each iteration.
    fLib = cell(1,iterNum);
    % Loop through the specified number of iterations.
    for p = 1:iterNum
        % Randomly select a set of points to estimate the model.
        sampleIdx = randIndex(ptNum, minPtNum);
        % Estimate the model using the randomly selected points.
        f1 = funcFindF(x(:, sampleIdx), y(:, sampleIdx));
        % Calculate the distances between the estimated model and all points.
        dist = funcDist(f1, x, y);
        % Find the indices of the inlier points based on the distance threshold.
        inlier1 = find(dist < thDist);
        % Store the number of inliers for this iteration.
        inlier_Num(p) = length(inlier1);
        % If the number of inliers is less than the threshold, skip to the next iteration.
        if length(inlier1) < Inlier_thres 
            continue; 
        end
        % Re - estimate the model using all the inlier points.
        fLib{p} = funcFindF(x(:, inlier1),y(:, inlier1));
    end
    % Find the index of the iteration with the maximum number of inliers.
    [~, idx] = max(inlier_Num);
    % Select the model estimated in the iteration with the most inliers.
    f = fLib{idx};
    % Calculate the distances between the final model and all points.
    dist = funcDist(f,x,y);
    % Find the indices of the inlier points for the final model.
    inlierIdx = find(dist < thDist);
end