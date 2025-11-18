% This function is used to find and draw the matched keypoints between two sets of keypoints.
% Input:
%   - kpts1: A matrix containing the information of keypoints in the first image. Each row represents a keypoint.
%   - kpts2: A matrix containing the information of keypoints in the second image. Each row represents a keypoint.
%   - descriptors1: A matrix containing the descriptors of keypoints in the first image. Each column corresponds to a keypoint's descriptor.
%   - descriptors2: A matrix containing the descriptors of keypoints in the second image. Each column corresponds to a keypoint's descriptor.
%   - threshold: A distance threshold used to determine if two keypoints are a match.
% Output:
%   - matched: A matrix where each row contains the indices of the matched keypoints (index in kpts1 and index in kpts2).
%   - locs1: A matrix containing the (x, y) coordinates of the matched keypoints in the first image.
%   - locs2: A matrix containing the (x, y) coordinates of the matched keypoints in the second image.
function [matched, locs1, locs2] = drawMatched(kpts1, kpts2, descriptors1, descriptors2, threshold)
    % Initialize the variable to store the indices of matched keypoints.
    matched = []; 
    % Initialize the variable to store the locations of matched keypoints in the first image.
    locs1 = [];
    % Initialize the variable to store the locations of matched keypoints in the second image.
    locs2 = [];
    % Initialize the variable to store the distances between matched descriptors.
    distance = [];
    % Loop through each descriptor in the first set.
    for kpt_i = 1:size(descriptors1, 2)
        % Loop through each descriptor in the second set.
        for kpt_j = 1:size(descriptors2, 2)
            % Check if the Euclidean distance between the two descriptors is less than the threshold.
            if (norm(descriptors1(:, kpt_i) - descriptors2(:, kpt_j), 2) < threshold)
                % If so, record the distance.
                distance = [distance, norm(descriptors1(:, kpt_i) - descriptors2(:, kpt_j), 2)];
                % And record the indices of the matched keypoints.
                matched = [matched; kpt_i, kpt_j];
            end
        end
    end
    % Draw matched correspondence
    % Loop through each pair of matched keypoints.
    for i = 1 : size(matched, 1)
        % Get the information of the matched keypoint in the first image.
        kpt1 = kpts1(matched(i, 1), :);
        % Get the information of the matched keypoint in the second image.
        kpt2 = kpts2(matched(i, 2), :);
        % Extract and record the (x, y) coordinates of the matched keypoint in the first image.
        locs1 = [locs1; kpt1(2), kpt1(1)];
        % Extract and record the (x, y) coordinates of the matched keypoint in the second image.
        locs2 = [locs2; kpt2(2), kpt2(1)];
    end  
end