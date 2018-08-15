function [J,Theta1, Theta2,b1,b2,w1,w2,w3,w4] = Gradient_Descent( X,y,Theta1,Theta2,lambda, ...
    Iterations, Alpha,b1,b2,w1,w2,w3,w4,offset )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
i=1;
Alpha = 0.00002;
while (i<100)
    
    [J(i),Theta1_grad,Theta2_grad,db1,db2,dw1,dw2,dw3,dw4] = nnCostFunction1(Theta1,Theta2,X,y,b1,b2,w1,w2,w3,w4,offset);
    Theta1 = Theta1 - Alpha*Theta1_grad;
    Theta2 = Theta2 - Alpha*Theta2_grad;
    b1 = b1 - Alpha*db1;
    b2 = b2 - Alpha*db2;
    w1 = w1 - Alpha*dw1;
    w2 = w2 - Alpha*dw2;
    w3 = w3 - Alpha*dw3;
    w4 = w4 - Alpha*dw4;
%     disp(sum(dw1(:)));
    disp('The cost function is');
    disp(J(i));
    disp(i);
    if(J(i)<0.1)
        p=0;
    end
    i= i+1;
    
end

end

