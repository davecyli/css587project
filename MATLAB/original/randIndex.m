% This function generates a set of unique random indices within a specified range.
% Input:
%   - maxIndex: The maximum index value from which to select.
%   - len: The number of random indices to generate.
% Output:
%   - index: A row vector containing the generated unique random indices.
function index = randIndex(maxIndex, len)
    % Check if the number of requested indices is greater than the maximum index.
    if len > maxIndex
        % If so, return an empty array.
        index = [];
        return
    end
    % Initialize an array to store the generated indices.
    index = zeros(1,len);
    % Create an array of available indices from 1 to maxIndex.
    available = 1:maxIndex;
    % Generate a set of random numbers within the appropriate range for each index selection.
    rs = ceil(rand(1,len).*(maxIndex:-1:maxIndex-len+1));
    % Loop through each index to be generated.
    for p = 1:len
        % Ensure the random number is non - zero.
        while rs(p) == 0
            rs(p) = ceil(rand(1)*(maxIndex-p+1));
        end
        % Select an index from the available indices based on the random number.
        index(p) = available(rs(p));
        % Remove the selected index from the available indices.
        available(rs(p)) = [];
    end
end