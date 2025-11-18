% This function stitches two images together based on their matched feature points.
% It also has an option to adjust the color of the images for a better seamless stitching result.
% Input:
%   - img1: The first input image.
%   - img2: The second input image.
%   - matchLoc1: A matrix containing the (x, y) coordinates of the matched feature points in img1.
%   - matchLoc2: A matrix containing the (x, y) coordinates of the matched feature points in img2.
%   - adjColor: An optional parameter. If it exists and is equal to 1, color adjustment will be performed.
% Output:
%   - output: The stitched image.
%   - H: The homography matrix that maps points from img2 to img1.
function [output,H] = stitch(img1, img2, matchLoc1, matchLoc2, adjColor)
    % Find the homography matrix that maps points from matchLoc2 to matchLoc1.
    % The points are transposed as the function findHomography expects columns to represent points.
    [H, corrPtIdx] = findHomography(matchLoc2', matchLoc1');
    % Create a transformation object for projective transformation using the homography matrix.
    T = maketform('projective', H'); 
    % Transform img2 using the projective transformation to align it with img1.
    img21 = imtransform(img2, T); 
    % Get the size of img1 (height, width, number of color channels).
    [M1, N1, dim] = size(img1);
    % Get the size of img2 (height, width, number of color channels).
    [M2, N2, ~] = size(img2);
    % Check if the adjColor parameter exists and is set to 1 for color adjustment.
    if exist('adjColor', 'var') && adjColor == 1
        % Set the radius for the local neighborhood around the matched points for color adjustment.
        radius = 2;
        % Extract the x and y coordinates of the corresponding matched points in img1.
        x1_corr = matchLoc1(corrPtIdx, 1);
        y1_corr = matchLoc1(corrPtIdx, 2);
        % Extract the x and y coordinates of the corresponding matched points in img2.
        x2_corr = matchLoc2(corrPtIdx, 1);
        y2_corr = matchLoc2(corrPtIdx, 2);
        % Get the number of corresponding matched points.
        corrPtLen = length(corrPtIdx);
        % Initialize arrays to store the sum of pixel values in the local neighborhoods for img1 and img2.
        s1 = zeros(1, corrPtLen);
        s2 = zeros(1, corrPtLen);
        % Loop through each color channel (for RGB images, dim is 3).
        for color = 1:dim
            % Loop through each corresponding matched point in img1 to calculate the sum of pixel values in its local neighborhood.
            for p = 1:corrPtLen
                % Calculate the left boundary of the local neighborhood.
                left = round(max(1, x1_corr(p)-radius));
                % Calculate the right boundary of the local neighborhood.
                right = round(min(N1, left+radius+1));
                % Calculate the upper boundary of the local neighborhood.
                up = round(max(1, y1_corr(p)-radius));
                % Calculate the lower boundary of the local neighborhood.
                down = round(min(M1, up+radius+1));
                % Calculate the sum of pixel values in the local neighborhood of img1 for the current color channel.
                s1(p) = sum(sum(img1(up:down, left:right, color)));
            end
            % Loop through each corresponding matched point in img2 to calculate the sum of pixel values in its local neighborhood.
            for p = 1:corrPtLen
                % Calculate the left boundary of the local neighborhood.
                left = round(max(1, x2_corr(p)-radius));
                % Calculate the right boundary of the local neighborhood.
                right = round(min(N2, left+radius+1));
                % Calculate the upper boundary of the local neighborhood.
                up = round(max(1, y2_corr(p)-radius));
                % Calculate the lower boundary of the local neighborhood.
                down = round(min(M2, up+radius+1));
                % Calculate the sum of pixel values in the local neighborhood of img2 for the current color channel.
                s2(p) = sum(sum(img2(up:down, left:right, color)));
            end
            % Calculate the scaling factor based on the size of the local neighborhood and the number of corresponding points.
            sc = (radius*2+1)^2 * corrPtLen;
            % Fit a linear polynomial to find the color adjustment coefficients.
            adjcoef = polyfit(s1/sc, s2/sc, 1);
            % Adjust the color of img1 using the calculated coefficients.
            img1(:,:,color) = img1(:, :, color) * adjcoef(1) + adjcoef(2);
        end
    end
    % Initialize a matrix to store the transformed corner points of img2.
    pt = zeros(3,4);
    % Transform the top-left corner point of img2.
    pt(:, 1) = H * [1; 1; 1];
    % Transform the top-right corner point of img2.
    pt(:, 2) = H * [N2; 1; 1];
    % Transform the bottom-right corner point of img2.
    pt(:, 3) = H * [N2; M2; 1];
    % Transform the bottom-left corner point of img2.
    pt(:, 4) = H * [1; M2; 1];
    % Convert the homogeneous coordinates of the transformed points to inhomogeneous coordinates for x and y.
    x2 = pt(1, :) ./ pt(3, :);
    y2 = pt(2, :) ./ pt(3, :);
    % Find the minimum y-coordinate among the transformed points.
    up = round(min(y2));
    % Initialize the vertical offset for the stitched image.
    Yoffset = 0;
    % If the minimum y-coordinate is less than or equal to 0, calculate the offset to shift the image up.
    if up <= 0
        Yoffset = -up + 1;
        up = 1;
    end
    % Find the minimum x-coordinate among the transformed points.
    left = round(min(x2));
    % Initialize the horizontal offset for the stitched image.
    Xoffset = 0;
    % If the minimum x-coordinate is less than or equal to 0, calculate the offset to shift the image left.
    if left<=0
        Xoffset = -left + 1;
        left = 1;
    end
    % Get the size of the transformed img2 (img21).
    [M3, N3, ~] = size(img21);
    % Place the transformed img2 (img21) into the output image at the appropriate position.
    output(up:up+M3-1, left:left+N3-1, :) = img21;
    % Place img1 into the output image at the appropriate position with the calculated offsets.
    output(Yoffset+1:Yoffset+M1, Xoffset+1:Xoffset+N1, :) = img1;
end