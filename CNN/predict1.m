function [ k ] = predict1( data, w1,w2,w3,w4,b1,b2,Theta1,Theta2,m )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
data1 = data;
conv1 = vl_nnconv(data1,w1,[]);                % After conv1 = 30,30,16  w1 = 3,3,3,16
% disp('max(conv1)');
relu1 = vl_nnrelu(conv1);                     %relu1 = 30,30,16
% disp('Size of relu1');
% disp(size(relu1));
conv2 = vl_nnconv(relu1,w2,[]);               %conv2 = 28,28,32  w2 = 3,3,3,32
relu2 = vl_nnrelu(conv2);                      %relu2 = 28,28,32
pool1 = vl_nnpool(relu2,13);                  % pool1 = 16,16,32
% meanpool1 = sum(pool1(:))/8192;
% diff1 = pool1-meanpool1;
% varx1 = (diff1.^2);
% varx = sum(varx1(:))/4096;
% norm1 = (diff1)/sqrt(varx);     %norm1 = 16,16,32
%disp(size(meanrelu1));

conv3 = vl_nnconv(pool1,w3,[]);           %conv3 = 14,14,64   w3=3,3,3,64
relu3 = vl_nnrelu(conv3);                     %relu3 = 14,14,64
conv4 = vl_nnconv(relu3,w4,[]);               %conv4 = 12,12,96     w4 = 3,3,3,96
relu4 = vl_nnrelu(conv4);                     %relu4 = 12,12,96
pool2 = vl_nnpool(relu4,7);                   %pool2 = 6,6,96
% meanpool2 = sum(pool2(:))/3456;
% varx2 = ((pool2-meanpool2).^2);
% varx2 = sum(varx2(:))/4096;
% G = (pool2-meanpool2)/sqrt(varx2);            %G = 6,6,96
X = reshape(pool2,3456,m);                        % X = 3456,1
%
% disp('max(X)');
% disp(max(X));
%disp(size(X));
%%
% disp(' Updated max(X)');
% disp(max(X));


J = 0;          % a1= 3456,1 Theta1= 500X3456 Theta2= 10X500
a1 = X;
z2 = Theta1*a1;    % z2 500,1
for i=1:m
    
    z2(:,i)= z2(:,i)+b1;   %b1= 500,1
end
a2 = relu(z2);      % a2= 500,1
z3 = Theta2*a2;        % z3= 10X1
for i= 1:m
    z3(:,i)= z3(:,i)+b2;   %b2= 10X1
end
a3 = relu(z3);     %a3= 10X1
h = a3;            %h=10*1
%disp(h);
k=0;
E = exp(h);
Sum = sum(E);
    for j = 1:10
        p(j,i)=exp(h(j))/Sum;
    end
    disp(p);

end

