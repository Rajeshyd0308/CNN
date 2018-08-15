function [J,Theta1_grad,Theta2_grad,db1,db2,dw1,dw2,dw3,dw4] = nnCostFunction1(Theta1,Theta2, ...
    data, labels,b1,b2,w1,w2,w3,w4,offest)
% Setup some useful variables

y = labels;
m = size(data, 4);
% loss= 0;
% dw1 = zeros(size(w1));
% dw2 = zeros(size(w2));
% dw3 = zeros(size(w3));
% dw4 = zeros(size(w4));
% dB1 = zeros(size(b1));
% dB2 = zeros(size(b2));
% Theta1_grad = zeros(size(Theta1));
% Theta2_grad = zeros(size(Theta2));
% J=0;
%---------------------convolution forward----------------------------------%
%32,32,3
% disp(max(data1));
% disp(max(w1));
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
h = a3;            %h=10,1
% disp('h');
% disp(h);
%% =============softmax=========================%%
yi = zeros(size(h));
for i= 1:m
    yi(y(i+offest),i)= 1;        % finally yi has shape 10,1
end
% %Softmax loss%
E = exp(h);
Sum = sum(E,1);
loss = 0;
for i=1:m
    T=exp(h(y(i+offest),i));
    l = -log(T/Sum(i));
    loss = loss+l;
    for j = 1:10
        p(j,i)=exp(h(j,i))/Sum(i);
    end
end
loss = loss/m;

%============================end softmax=================================%
%--------Regularization-----------%
% S = 0;
% K= 0;
% w1r = w1.^2;
% w2r = w2.^2;
% Theta1r = Theta1.^2;
% Theta2r = Theta2.^2;
% S = sum(Theta1r(:));
% K = sum(Theta2r(:));
% w1s = sum(w1r(:));
% w2s = sum(w2r(:));
% S = S/m;
% K = K/m;
% w1s = w1s/m;
% w2s = w2s/m;
% R = S+K+w1s+w2s;
% R = lambda*(R/2);
% J = loss + R;

% =========================================================================
%------------Back Prop------------%
dp = p-yi;                    %d3 10X1  error in output
% disp(dp);
% Chain rule
dz3 = backrelu(a3).*dp;    % 10X1
dTheta2 = (dz3*a2');             % 10X25
da2 = Theta2'*dz3;                 % 25X4000
dz2 = backrelu(a2).*da2;
dTheta1 = (dz2*X');             % 500X3456
dB1 = sum(dz2,2);               % Summing over all the elements of a row
dB2 = sum(dz3,2);
dX = (Theta1')*dz2;
dX = reshape(dX,6,6,96,m);
%disp(sum(dX(:)));
dpool = vl_nnpool(relu4,7,dX);
%disp('dpool is ');
%disp(sum(dpool(:)));
drelu4 = vl_nnrelu(conv4,dpool);
[dconv4,dw4,DB] = vl_nnconv(relu3,w4,[],drelu4);
drelu3 = vl_nnrelu(conv3,dconv4);
[dconv3,dw3,DB] = vl_nnconv(pool1,w3,[],drelu3);
dpool2 = vl_nnpool(relu2,13,dconv3);
drelu2 = vl_nnrelu(conv2,dpool2);
[dconv2,dw2,DB] = vl_nnconv(relu1,w2,[],drelu2);
drelu1 = vl_nnrelu(conv1,dconv2);
[dconv1,dw1,DB] = vl_nnconv(data1,w1,[],drelu1);

dw1 = dw1/m;
dw2 = dw2/m;
dw3 = dw3/m;
dw4 = dw4/m;
Theta1_grad=dTheta1/m;
Theta2_grad=dTheta2/m;
db1= dB1/m;
db2= dB2/m;
J= loss;

end
