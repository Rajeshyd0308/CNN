
function [J,Theta1_grad,Theta2_grad,dB1,dB2,dw1,dw2] = nnCostFunction(Theta1,Theta2, ...
    data, labels, lambda,b1,b2,w1,w2,num_conv1,num_conv2,num_pool)
% Setup some useful variables
y = labels;
m = size(data, 4);
%---------------------convolution forward----------------------------------%

conv1 = vl_nnconv(data,w1,[]);
relu1 = vl_nnrelu(conv1);
conv2 = vl_nnconv(relu1,w2,[]);
relu2 = vl_nnrelu(conv2);
G = vl_nnpool(relu2,num_pool);
X = reshape(G,4096,1000);
%%

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));          % X 400X4000 Theta1 25X400 Theta2 10X25
a1 = X;
z2 = Theta1*a1;    % z2 25X4000
for i=1:m
    z2(:,i)= z2(:,i)+b1;   %b1   25X1 and z2 shape unchanged
end
a2 = relu(z2);      % a2 25X4000
z3 = Theta2*a2;        % z3 10X4000
for i=1:m
    z3(:,i)= z3(:,i)+b2;   %b2 10X1
end
a3 = relu(z3);     % 10X4000
h = a3;
yi = zeros(size(h));
for i =1:m
    yi(y(i),i)= 1;        % finally yi has shape 10X4000
end
% %Softmax loss%

E = exp(h);
Sum = sum(E,1);
loss = 0;
%Vectorization of this part is possible if column wise division operator available %
for i=1:m
    T=exp(h(y(i),i));
    l = -log(T/Sum(i));
    loss = loss+l;
    for j = 1:10
        p(j,i)=exp(h(j,i))/Sum(i);
    end
end
loss = loss/m;
disp('loss');
disp(loss);
J=loss;
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
%------------Back Prop------------%

dp = p-yi;                    %d3 10X4000  error in output
% Chain rule
dz3 = backrelu(a3).*dp;    % 10X4000
dTheta2 = (dz3*a2');             % 10X25
da2 = Theta2'*dz3;                 % 25X4000
dz2 = backrelu(a2).*da2;
dTheta1 = (dz2*X');             % 25X400
dB1 = sum(dz2,2)/m;               % Summing over all the elements of a row
dB2 = sum(dz3,2)/m;
Theta1_grad=dTheta1+((lambda/m)*Theta1);
Theta2_grad=dTheta2+((lambda/m)*Theta2);
dX = (Theta1')*dz2;
dX = reshape(dX,16,16,16,1000);
dpool = vl_nnpool(relu2,num_pool,dX);
disp('size of dpool');
drelu2 = vl_nnrelu(conv2,dpool);
disp(size(drelu2));
[dconv2,dw2] = vl_nnconv(relu1,w2,[],drelu2);
dw2 = dw2 + (lambda/m)*w2;
drelu1 = vl_nnrelu(conv1,dconv2);
[dconv1,dw1] = vl_nnconv(data,w1,[],drelu1);
dw1 = dw1 + (lambda/m)*w1;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
%grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
