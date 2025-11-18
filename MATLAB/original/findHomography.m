% This function is used to find the homography matrix between two sets of points using the RANSAC algorithm.
% Input:
%   - pts1: A matrix of points in the first image. Each column represents a point in the form [x; y].
%   - pts2: A matrix of corresponding points in the second image. Each column represents a point in the form [x; y].
% Output:
%   - H: The computed homography matrix that maps points from pts1 to pts2.
%   - corrPtIdx: The indices of the inlier points that are used to compute the final homography matrix.
function [H, corrPtIdx] = findHomography(pts1, pts2)
    % Set the minimum number of points required to compute the homography matrix.
    % In the case of homography, at least 4 non - collinear points are needed.
    coef.minPtNum = 4; 
    % Set the number of iterations for the RANSAC algorithm.
    % More iterations generally lead to a more accurate result but also increase the computational time.
    coef.iterNum = 30; 
    % Set the distance threshold for determining inlier points.
    % Points whose reprojection error is less than this threshold are considered inliers.
    coef.thDist = 4; 
    % Set the threshold for the inlier ratio.
    % If the ratio of inliers to the total number of points is less than this value, the result may be unreliable.
    coef.thInlrRatio = 0.1;
    % Call the RANSAC function to compute the homography matrix.
    % The @solveHomography is a handle to the function that computes the homography matrix from a set of point correspondences.
    % The @calcDist is a handle to the function that calculates the distance (reprojection error) between the transformed points and the actual points.
    [H, corrPtIdx] = RANSAC(pts1, pts2, coef, @solveHomography, @calcDist);
end