% This function tests and finds the feature points in an image using a specific method.
% It first divides the image into blocks and then determines the extreme points (maximum and minimum) in each block.
% Input:
%   - ori_image: The original input image (used for additional checks to confirm extreme points).
%   - input_img: The image on which feature points are to be detected. It will be converted to grayscale internally.
%   - q: A vector containing values that are used to determine the size of the blocks for dividing the image.
% Output:
%   - all_extremal_point_position: A matrix where each row represents an extreme feature point with the following information:
%       - Column 1: The row position of the feature point in the image.
%       - Column 2: The column position of the feature point in the image.
%       - Column 3: The value of the feature point in the original image (ori_image).
%       - Column 4: The logarithm (base 2) of the block size (related to the value from q).
%       - Column 5: The row index of the block to which the feature point belongs.
%       - Column 6: The column index of the block to which the feature point belongs.

function all_extremal_point_position = Test_feature_points_DE(ori_image,input_img,q)
    % Convert the input image to grayscale for further processing.
    image = img2gray(input_img);
    
    % Get the number of elements in the q vector.
    [~,n_q] = size(q);
    
    % Get the height (m) and width (n) of the grayscale image.
    [m,n] = size(image);
    
    % Initialize a vector to store the number of feature points expected for each block size in q.
    conut_feature_point = zeros(1,n_q);
    % Calculate the number of feature points for each block size in q.
    for i = 1:n_q
        s = q(i);
        conut_feature_point(i) = 2*floor(m/s)*floor(n/s);
    end
    % Calculate the total number of feature points.
    number_feature = sum(conut_feature_point);
    % Initialize a matrix to store all the extreme feature points.
    all_extremal_point_position = zeros(number_feature,6);
    % Initialize a variable to keep track of the position in the all_extremal_point_position matrix.
    middle_point = 0;
    % Loop through each block size in q.
    for ii = 1:n_q
        s = q(ii);
        % Calculate the number of rows of blocks in the image.
        row = floor(m/s);
        % Calculate the number of columns of blocks in the image.
        col = floor(n/s); 
        % Initialize matrices to store the maximum value in each block.
        max_value_block = zeros(row,col);
        % Initialize matrices to store the row position of the maximum value in each block.
        max_position_k = zeros(row,col);
        % Initialize matrices to store the column position of the maximum value in each block.
        max_position_r = zeros(row,col);
        % Initialize matrices to store the minimum value in each block.
        min_value_block = zeros(row,col);
        % Initialize matrices to store the row position of the minimum value in each block.
        min_position_k = zeros(row,col);
        % Initialize matrices to store the column position of the minimum value in each block.
        min_position_r = zeros(row,col);
        % Loop through each block in the image.
        for i = 1:row
            for j = 1:col
                % Extract the current block from the grayscale image.
                block = image((i-1)*s+1:i*s,(j-1)*s+1:j*s);   
                % Find the maximum value in the block.
                max_value_block(i,j) = max(max(block)); 
                % Find the position (row and column) of the maximum value in the block.
                [max_k,max_r] = find(max_value_block(i,j) == block);
                % Calculate the global row position of the maximum value in the image.
                max_position_k(i,j) = max_k(1)+(i-1)*s; 
                % Calculate the global column position of the maximum value in the image.
                max_position_r(i,j) = max_r(1)+(j-1)*s;

                % Find the minimum value in the block.
                min_value_block(i,j) = min(min(block));
                % Find the position (row and column) of the minimum value in the block.
                [min_k,min_r] = find(min_value_block(i,j) == block);
                % Calculate the global row position of the minimum value in the image.
                min_position_k(i,j) = min_k(1)+(i-1)*s;
                % Calculate the global column position of the minimum value in the image.
                min_position_r(i,j) = min_r(1)+(j-1)*s;
            end
        end

        % Initialize a matrix to store the position and value of feature points (both maximum and minimum).
        feature_point_position_value = zeros(2*row*col,6);
        % Fill the matrix with information about the maximum feature points in each block.
        for i = 1:row
            for j = 1:col
                feature_point_position_value((i-1)*col+j,1) = max_position_k(i,j);
                feature_point_position_value((i-1)*col+j,2) = max_position_r(i,j);
                feature_point_position_value((i-1)*col+j,3) = max_value_block(i,j);
                feature_point_position_value((i-1)*col+j,5) = i;
                feature_point_position_value((i-1)*col+j,6) = j;
                % Fill the matrix with information about the minimum feature points in each block.
                feature_point_position_value(row*col+(i-1)*col+j,1) = min_position_k(i,j);
                feature_point_position_value(row*col+(i-1)*col+j,2) = min_position_r(i,j);
                feature_point_position_value(row*col+(i-1)*col+j,3) = min_value_block(i,j);
                feature_point_position_value(row*col+(i-1)*col+j,5) = i;
                feature_point_position_value(row*col+(i-1)*col+j,6) = j;
            end
        end
        % Initialize a matrix to store the extreme feature points with additional information.
        extremal_point_position_value = zeros(2*row*col,6);
        % Copy the position information from feature_point_position_value.
        extremal_point_position_value(:,1:2) = feature_point_position_value(:,1:2);
        % Set the fourth column to the logarithm (base 2) of the block size.
        extremal_point_position_value(:,4) = log2(s);
        % Copy the block index information from feature_point_position_value.
        extremal_point_position_value(:,5:6) = feature_point_position_value(:,5:6);
        % Initialize a matrix to store the selected extreme feature points.
        select_extremal_point_position_value = zeros(2*row*col,6);
        % Set the third column of extremal_point_position_value to the value from the original image.
        for i = 1:2*row*col
            extremal_point_position_value(i,3) = ori_image(feature_point_position_value(i,1),feature_point_position_value(i,2));
        end

        % Loop through each potential extreme feature point.
        for i = 1:2*row*col
            % Determine the starting row index for the neighborhood check.
            if extremal_point_position_value(i,1)-1 <= 0
                begin_k = 1;
            else
                begin_k = extremal_point_position_value(i,1)-1;
            end

            % Determine the ending row index for the neighborhood check.
            if extremal_point_position_value(i,1)+1 > m
                last_k = m;
            else
                last_k = extremal_point_position_value(i,1)+1;
            end

            % Determine the starting column index for the neighborhood check.
            if extremal_point_position_value(i,2)-1 <= 0
                begin_r = 1;
            else
                begin_r = extremal_point_position_value(i,2)-1;
            end

            % Determine the ending column index for the neighborhood check.
            if extremal_point_position_value(i,2)+1 > n
                last_r = n;
            else
                last_r = extremal_point_position_value(i,2)+1;
            end

            % Extract the neighborhood around the potential feature point from the original image.
            temp = ori_image(begin_k:last_k,begin_r:last_r);
            % Count the number of pixels with the same value as the potential feature point in the neighborhood.
            a = length(find(temp==extremal_point_position_value(i,3)));
            % If there is only one such pixel, consider it as a valid extreme feature point.
            if a == 1
                select_extremal_point_position_value(i,:) = extremal_point_position_value(i,:);
            end
        end
        % Store the selected extreme feature points in the all_extremal_point_position matrix.
        all_extremal_point_position(middle_point+1:middle_point+conut_feature_point(ii),:) = select_extremal_point_position_value;
        % Update the position in the all_extremal_point_position matrix.
        middle_point = middle_point+conut_feature_point(ii);
    end
    % Remove rows with all zero values (if any) from the all_extremal_point_position matrix.
    all_extremal_point_position(all(all_extremal_point_position == 0,2),:) = [];
    % Get the size of the matrix containing the final extreme feature points.
    [m_feture_point_position,~] = size(all_extremal_point_position);
    % Create a message string to display the number of found feature points.
    aaaa = strcat('find ',num2str(m_feture_point_position),' keypoints!!!');
    % Display the message.
    disp(aaaa);
end