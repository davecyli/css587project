% This function computes descriptors for a set of keypoints in an input image.
% Input:
%   - input_img: The input image for which descriptors are to be computed.
%   - all_extremal_point_position: A matrix containing the positions and other 
%     information of all the keypoints. Each row represents a keypoint with 
%     multiple columns of associated data.
% Output:
%   - descriptors: A matrix where each column is a descriptor vector for a keypoint.
function descriptors = Compute_descriptors(input_img,all_extremal_point_position)
    % Convert the input image to grayscale if it's a color image.
    img = img2gray(input_img);
    % d represents the number of sub - regions in each dimension of the descriptor block.
    d = 4; 
    % n represents the number of bins in each histogram within a sub - region.
    n = 4;   
    % The total number of elements in a descriptor vector is 4 * 4 * 4 = 64.
    % Initialize the descriptors matrix with zeros, with 64 rows and the number 
    % of columns equal to the number of keypoints.
    descriptors = zeros(64,size(all_extremal_point_position, 1));
    % Loop through each keypoint.
    for kpt_i = 1:size(all_extremal_point_position, 1)
        % Extract the information of the current keypoint.
        kpt = all_extremal_point_position(kpt_i, :);
        % Extract the scale information of the keypoint.
        scale = kpt(4);
        % Calculate the scaled scale factor.
        s_scale = 2^kpt(4);
        % Extract the block indices of the keypoint.
        block_i = kpt(5);
        block_j = kpt(6);
        % Calculate the adjusted height and width of the keypoint within the block.
        kpt_h = kpt(1)-s_scale*(block_i-1);
        kpt_w = kpt(2)-s_scale*(block_j-1);
        
        % Extract the relevant block from the grayscale image.
        G_pyramid_i = img((block_i-1)*s_scale+1:block_i*s_scale,(block_j-1)*s_scale+1:block_j*s_scale);
        % Get the height and width of the extracted block.
        [G_pyramid_i_h, G_pyramid_i_w] = size(G_pyramid_i);

        % Calculate the width of the histogram.
        hist_width = 3*sqrt(scale); 
        % Calculate the radius of the neighborhood around the keypoint.
        radius = round(hist_width * d); 
        % Ensure the radius does not exceed the maximum possible value within the block.
        radius = min(radius, floor(sqrt(G_pyramid_i_h^2 + G_pyramid_i_w^2))); 
        % Initialize the descriptor vector for the current keypoint.
        descriptor_i = zeros(d*d*n,1);
        % Loop through each sub - region in the x - dimension of the descriptor block.
        for ii = 1:d
            % Loop through each sub - region in the y - dimension of the descriptor block.
            for jj = 1:d
                % Initialize the histogram for the current sub - region.
                hist = zeros(1,n);
                % Loop through each pixel in the x - direction within the sub - region.
                for i = -radius/d:radius/d
                    % Loop through each pixel in the y - direction within the sub - region.
                    for j = -radius/d:radius/d
                        % Calculate the height and width of the pixel in the image.
                        img_h = floor(kpt_h + i + (ii-1)*radius/d);
                        img_w = floor(kpt_w + j + (jj-1)*radius/d);
                        
                        % Check if the pixel is within the valid range of the block.
                        if img_h>1 && img_h<G_pyramid_i_h && img_w>1 && img_w<G_pyramid_i_w
                            % Calculate the gradient in the x - direction.
                            dx = double(G_pyramid_i(img_h, img_w + 1) - G_pyramid_i(img_h, img_w - 1));
                            % Calculate the gradient in the y - direction.
                            dy = double(G_pyramid_i(img_h - 1, img_w) - G_pyramid_i(img_h + 1, img_w));
                            
                            % Update the histogram based on the x - gradient.
                            if dx>=0
                                hist(1) = hist(1)+dx;
                            else
                                hist(3) = hist(3)+dx;
                            end

                            % Update the histogram based on the y - gradient.
                            if dy>=0
                                hist(2) = hist(2)+dy;
                            else
                                hist(4) = hist(4)+dy;
                            end
                        end
                    end 
                end
                % Place the histogram values into the appropriate position in the descriptor vector.
                descriptor_i(((ii-1)*d+jj-1)*n+1:((ii-1)*d+jj)*n) = hist;
            end
        end
        
        % Calculate the L2 - norm of the descriptor vector.
        descriptor_norm = norm(descriptor_i, 2);
        % Calculate the threshold for the descriptor vector.
        descriptor_threshold = 0.2 * descriptor_norm;
        % Threshold the descriptor vector to limit large values.
        descriptor_i(descriptor_i > descriptor_threshold) = descriptor_threshold;

        % Recalculate the L2 - norm of the thresholded descriptor vector.
        descriptor_norm = norm(descriptor_i, 2);
        % Normalize the descriptor vector.
        descriptor_i = descriptor_i ./ descriptor_norm;
        % Place the descriptor vector for the current keypoint into the descriptors matrix.
        descriptors(:,kpt_i) = descriptor_i;
    end
end