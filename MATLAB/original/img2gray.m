% This function is used to convert an input image to a grayscale image.
% Input:
%   - im: An input image, which can be either a color image (3 - dimensional matrix) or a grayscale image (2 - dimensional matrix).
% Output:
%   - gray: The resulting grayscale image.
function gray=img2gray(im)
    % Check the number of dimensions of the input image.
    % If the length of the size array is 3, it means the input is a color image.
    if length(size(im))==3
        % Convert the color image to a grayscale image using the rgb2gray function.
        gray=rgb2gray(im);
    else
        % If the input is not a color image (i.e., it's already a grayscale image), 
        % just assign the input image to the output variable.
        gray=im;
    end