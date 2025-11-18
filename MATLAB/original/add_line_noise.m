% This function is used to add line noise to a grayscale image.
% Input:
%   - gray_img: A grayscale image represented as a matrix.
% Output:
%   - distimg: The grayscale image with added line noise.
function distimg = add_line_noise(gray_img)
    % Get the size of the grayscale image. m_gray_img is the number of rows,
    % and n_gray_img is the number of columns.
    [m_gray_img,n_gray_img] = size(gray_img);
    % Generate a vector of noise values. The length of the vector is equal to 
    % the total number of pixels in the grayscale image. The noise values increase 
    % linearly from 1e - 10 to 1e - 10 times the total number of pixels.
    length_noise = 1e-10*(1:m_gray_img*n_gray_img);
    % Reshape the noise vector into a matrix with n_gray_img rows and m_gray_img columns.
    noise_1 = reshape(length_noise,[n_gray_img,m_gray_img]);
    % Transpose the noise matrix to match the dimensions of the grayscale image.
    noise = noise_1';
    % Add the noise matrix to the grayscale image to obtain the image with added noise.
    distimg=gray_img+noise;
end