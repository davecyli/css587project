% This function calculates the distance (reprojection error) between the transformed points and the actual points.
% Input:
%   - H: The homography matrix used to transform the points from pts1 to pts2.
%   - pts1: A matrix of points in the first image. Each column represents a point in the form [x; y].
%   - pts2: A matrix of corresponding points in the second image. Each column represents a point in the form [x; y].
% Output:
%   - dis: A row vector containing the squared Euclidean distances between the transformed points and the actual points for each point pair.
function dis = calcDist(H, pts1, pts2)
    % Get the number of points in pts1.
    n = size(pts1, 2);
    % Homogeneous transformation of pts1 using the homography matrix H.
    % Append a row of ones to pts1 to convert it to homogeneous coordinates.
    % Then multiply it by the homography matrix H.
    pts3 = H * [pts1; ones(1,n)];
    % Convert the transformed points back to inhomogeneous coordinates.
    % Divide the first two rows of pts3 by the third row.
    pts3 = pts3(1:2,:) ./ repmat(pts3(3,:), 2, 1);
    % Calculate the squared Euclidean distance between the transformed points (pts3) and the actual points (pts2).
    % Sum the squared differences along each column.
    dis = sum((pts2-pts3).^2, 1);
end