% function reads in two matrices of points and calculates distance between each
% corresponding set of points
% returns array of errors
function [error] = calculateError(points1, points2)
    length = size(points1, 1);
    errors = zeros(1, length);
    for i=1:length
        errors(i) = pdist([points1(i); points2(i)], 'euclidean');
    end
    error = mean(errors);

    