clear all; close all; clc;
X = load('mv_regressionx.dat'); Y = load('mv_regressiony.dat');
N = length(X);
X = normalize(X);
X = [ones(N,1), X];


% Multivariate stochastic gradient descent
%alpha=0.01;
[rows,cols] = size(X);

iterations = 1500;

A_history = zeros(N,1);
alpha = 0.01;
step = 0.15
while alpha < 1.4
    alpha
    theta = rand(cols, 1);
    for num_iterations = 1:iterations

        result = zeros(cols,1);
        cost = X * theta - Y;
        for t = 1:cols
            result(t) = sum(cost .* X(:,t));
        end
        % update
        theta = theta - alpha * (1/N) * result;

        %theta = theta - (alpha/N) * (X' * (X * theta - Y))
    end
    A_history = [X * theta, A_history];
    alpha = alpha + step;
end
A_history = A_history(:,1:end-1);

plot(A_history(:,1), '-')
hold on
plot(A_history(:,8), '-')
%figure
%hold on
%scatter(df(:,2),df(:,4))
%plot(df(:,2), theta(1) + theta(2) * df(:,2) + theta(3) * df(:,3) , '-')