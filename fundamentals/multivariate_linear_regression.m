clear all; close all; clc;
X = load('mv_regressionx.dat'); Y = load('mv_regressiony.dat');
N = length(X);
X = normalize(X)
X = [ones(N,1), X];


% Multivariate stochastic gradient descent
alpha=1.3;
theta = rand(3, 1);

iterations = 1500;

for num_iterations = 1:iterations

    % y- prediction
    %y_current = theta(1) + theta(2) .* X(:,3);
    % current cost of prediction
    %cost_current = y_current - Y(:,1)
    %j1=(1/N)*sum(cost_current);
    %j2=(1/N)*sum((cost_current).*X(:,3));
    % update theta
    %theta(1)=theta(1)-alpha*(j1);
    %theta(2)=theta(2)-alpha*(j2); 
    
    theta = theta - (alpha/N) * (X' * (X * theta - Y))
 
end
plot(X * theta)

%figure
%hold on
%scatter(df(:,2),df(:,4))
%plot(df(:,2), theta(1) + theta(2) * df(:,2) + theta(3) * df(:,3) , '-')