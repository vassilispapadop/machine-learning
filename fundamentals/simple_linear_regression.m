% Simple Linear Regression

% The dataset consists of CDC growth statistics for boys
%
% x= age (years)
% y= height (meters)
%

clear all; close all; clc;
x = load('linear_regressionx.dat'); y = load('linear_regressiony.dat');

m = length(y); % number of data points = 50


% Visualize the training data
figure;
plot(x, y, 'o');
title({'Training Data'})
ylabel('Y: Height (meters)')
xlabel('X: Age (years)')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 1. Using 'fitlm' function: "fitlm(x,y)" uses intercept by default 2 
a1 = fitlm(x,y);
W=a1.Coefficients{:,1}

 
% Visualize results
figure
hold on
scatter(x,y)
plot(x, W(1)+W(2)*x, '-')
ylabel('Y: Height (meters)')
xlabel('X: Age (years)')
title({'Simple Linear Regression using fitlm(x,y) function'});
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 2. Using 'regress' function: "regress(y,x)" uses no intercept by default. 
% We could add intercept by adding "ones" matrix.

% To compute coefficient estimates for a model with a constant term (intercept), include a column of ones in the matrix X.
x2 = [ones(size(x)), x];
a2 = regress(y,x2);

% Visualize results
figure
hold on
scatter(x,y)
plot(x, a2(1)+a2(2)*x, '-')
ylabel('Y: Height (meters)')
xlabel('X: Age (years)')
title({'Simple Linear Regression using regress(x,y) function'});
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 3. Using Stochastic Gradient Descent
alpha=0.07;
theta = zeros(2,1);
iterations = 1500;
w_current =0;
b_current = 0;
for num_iterations = 1:iterations
    k=1:m;
    % y- prediction
    y_current = theta(1) + theta(2) .* x2(k,2);
    % current cost of prediction
    cost_current = y_current - y(k);
    j1=(1/m)*sum(cost_current);
    j2=(1/m)*sum((cost_current).*x2(k,2));
    % update theta
    theta(1)=theta(1)-alpha*(j1);
    theta(2)=theta(2)-alpha*(j2);  
end


% Visualize results
figure
hold on
scatter(x,y)
plot(x, theta(1) + theta(2) * x, '-')

ylabel('Y: Height (meters)')
xlabel('X: Age (years)')
title({'Simple Linear Regression using SGD method'});
hold off



