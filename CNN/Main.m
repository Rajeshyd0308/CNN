
%% Initialization
clear ; close all; clc

% num_conv1 = [64 2 1]; %number of filters, padding, stride
% num_conv2 = [16 2 1];
% num_pool = 13;
%Fully connected layer%

input_layer_size  = 3456;  
hidden_layer_size = 500;   
num_labels = 10;         
                          
 


% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('data_batch_1.mat');
data = data(1:20,:);
offset =0;
% mean = sum(data(:))/(3072*100);
% data = data-mean;
m = size(data, 1);
disp(m);
data = im2double(data);
data = data';
data = reshape(data,32,32,3,m);
% Randomly select 12 data points to display
labels = labels +1;


fprintf('\nInitializing Neural Network Parameters ...\n')
w1 = double(randn(3,3,3,16,'single')/sqrt(27));
w2 = double(randn(3,3,16,32,'single')/sqrt(144));
w3 = double(randn(3,3,32,64,'single')/sqrt(288));
w4 = double(randn(3,3,64,96,'single')/sqrt(576));
Theta1 = double(randn(hidden_layer_size,input_layer_size)/sqrt(3456));
Theta2 = double(randn(num_labels, hidden_layer_size)/sqrt(500));
b1 = double(randn(hidden_layer_size,1)/sqrt(500));
b2 = double(randn(num_labels,1)/sqrt(10));


% J = nnCostFunction1(Theta1,Theta2, ...
%     data, labels,b1,b2,w1,w2,w3,w4,num_conv1,num_conv2,num_pool);



fprintf('\nTraining Neural Network... \n')


lambda = 1;
Iterations = 1000;
Alpha = 2;
[J,Theta1,Theta2,b1,b2,w1,w2,w3,w4] = Momentum_Update(data,labels,Theta1,Theta2,lambda,Iterations,Alpha,b1,b2,w1,w2,w3,w4,offset);

fprintf('Program done\n');
% pause;
% 
% 
% 
% pred = predict(Theta1, Theta2, Xtest,b1,b2);
% 
% fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);
% P = Xtest(1,:);
% num = numfind(P,Theta1,Theta2,b1,b2);
% disp(num);


