function [J,Theta1, Theta2,b1,b2,w1,w2,w3,w4] = Momentum_Update(  X,y,Theta1,Theta2,lambda, ...
    Iterations, Alpha,b1,b2,w1,w2,w3,w4,offset )
%Nestrov Momentum Update
i=1;
mu = 0.9;
v1=zeros(size(Theta1));
v2=zeros(size(Theta2));
v3=zeros(size(b1));
v4=zeros(size(b2));
v5=zeros(size(w1));
v6=zeros(size(w2));
v7 = zeros(size(w3));
v8 = zeros(size(w4));
Alpha = 0.0001;
while (i<20)
    Theta1 = Theta1 + mu*v1;
    Theta2 = Theta2 + mu*v2;
    b1 = b1 + mu*v3;
    b2 = b2 + mu*v4;
    w1 = w1 + mu*v5;
    w2 = w2 + mu*v6;
    w3 = w3 + mu*v7;
    w4 = w4 + mu*v8;
    [J(i),Theta1_grad,Theta2_grad,db1,db2,dw1,dw2,dw3,dw4] = nnCostFunction1(Theta1,Theta2,X,y,b1,b2,w1,w2,w3,w4,offset);
    v1 = mu*v1 - Alpha*Theta1_grad;
    v2 = mu*v2 - Alpha*Theta2_grad;
    v3 = mu*v3 - Alpha*db1;
    v4 = mu*v4 - Alpha*db2;
    v5 = mu*v5 - Alpha*dw1;
    v6 = mu*v6 - Alpha*dw2;
    v7 = mu*v7 - Alpha*dw3;
    v8 = mu*v8 - Alpha*dw4;
    Theta1 = Theta1 +v1;
    Theta2 = Theta2 +v2;
    b1 = b1 + v3;
    b2 = b2 + v4;
    w1 = w1 + v5;
    w2 = w2 + v6;
    w3 = w3 + v7;
    w4 = w4 + v8;
    disp('The cost function is');
    disp(J(i));
    disp(i);
    i= i+1;
    
end

end

