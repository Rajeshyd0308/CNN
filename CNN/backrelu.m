function [ k ] = backrelu( q )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
[c,v] = size(q);
k = zeros(c,v);
for i = 1:c
    for j = 1:v
        if q(i,j)>0
        k(i,j)= 1;
        end
    end
end
end

