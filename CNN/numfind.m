function [ j ] = numfind( p,Theta1,Theta2,b1,b2 )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
l1 = p';
q = Theta1*l1;
q = q+b1;
l2 = relu(q);
d = Theta2*l2;
d = d+b2;
l3 = relu(d);
[dummy,j] = max(l3);

end

