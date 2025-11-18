% This function computes the homography matrix H that maps a set of points pts1 to another set of points pts2.
% The homography matrix is a 3x3 matrix used in computer vision for tasks such as image stitching and perspective transformation.
% Input:
%   - pts1: A 2-by-n matrix where each column represents a point [x; y] in the first image or coordinate system.
%   - pts2: A 2-by-n matrix where each column represents the corresponding point [x; y] in the second image or coordinate system.
% Output:
%   - H: A 3x3 homography matrix that maps points from pts1 to pts2.

function H = solveHomography(pts1, pts2)
    % Get the number of point correspondences.
    n = size(pts1, 2);
    % Initialize a 2*n by 9 matrix A to build the linear system for solving the homography.
    A = zeros(2*n, 9);

    % Fill the first part of matrix A. For odd rows (related to x-coordinate equations),
    % set the first two columns to the (x, y) coordinates of pts1 and the third column to 1.
    A(1:2:2*n, 1:2) = pts1';
    A(1:2:2*n, 3) = 1;
    % For even rows (related to y-coordinate equations),
    % set the fourth and fifth columns to the (x, y) coordinates of pts1 and the sixth column to 1.
    A(2:2:2*n, 4:5) = pts1';
    A(2:2:2*n, 6) = 1;

    % Extract the x and y coordinates of pts1 and pts2 as column vectors.
    x1 = pts1(1, :)';
    y1 = pts1(2, :)';
    x2 = pts2(1, :)';
    y2 = pts2(2, :)';

    % Fill the remaining part of matrix A based on the homography equations.
    % For odd rows, set the seventh column to -x2 * x1, the eighth column to -x2 * y1, and the ninth column to -x2.
    A(1:2:2*n, 7) = -x2 .* x1;
    A(1:2:2*n, 8) = -x2 .* y1;
    A(1:2:2*n, 9) = -x2;
    % For even rows, set the seventh column to -y2 * x1, the eighth column to -y2 * y1, and the ninth column to -y2.
    A(2:2:2*n, 7) = -y2 .* x1;
    A(2:2:2*n, 8) = -y2 .* y1;
    A(2:2:2*n, 9) = -y2;

    % Compute the eigenvectors and eigenvalues of A' * A.
    % We are interested in the eigenvector corresponding to the smallest eigenvalue.
    [evec, ~] = eig(A' * A);

    % Reshape the first eigenvector (corresponding to the smallest eigenvalue) into a 3x3 matrix.
    % Then transpose it to get the correct orientation of the homography matrix.
    H = reshape(evec(:, 1),[3, 3])';

    % Normalize the homography matrix by dividing all elements by the last element (H(3, 3)).
    % This is done to ensure a unique representation of the homography matrix.
    H = H / H(end); 
end