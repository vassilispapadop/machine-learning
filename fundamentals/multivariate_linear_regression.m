clear all; close all; clc;
X = load('mv_regressionx.dat'); Y = load('mv_regressiony.dat');
N = length(X);
X = normalize(X);
X = [ones(N,1), X];


% Multivariate stochastic gradient descent
%alpha=0.01;
[rows,cols] = size(X);

iterations = 100;

alpha_values = linspace(.01, 1.3, 10);
[a_rows,a_cols] = size(alpha_values);
Alpha_history = zeros(iterations,a_cols);

index = 1;
for alpha = alpha_values
    disp(['alpha is: [' num2str(alpha) ']']) ;
    % initialize with random numbers
    theta = rand(cols, 1);
    SquaredErrors_hist = zeros(iterations,1);
    for iter = 1:iterations

        result = zeros(cols,1);
        cost = X * theta - Y;
        for t = 1:cols
            result(t) = sum(cost .* X(:,t));
        end
        % update
        theta = theta - alpha * (1/N) * result;
        % average squared-errors
        J = sum((X*theta - Y).^2)/(2*N);
        SquaredErrors_hist(iter) = J;
        
        Alpha_history(iter, index) = SquaredErrors_hist(iter);

    end
    index = index + 1;
end

colNames = cell(a_cols);
i = 1;
for alpha = alpha_values
    colNames{i} = num2str(alpha);
    i = i + 1;
end
Alpha_history = array2table(Alpha_history, 'VariableNames',colNames(:,1));
plot(Alpha_history, '-') 
%i = 1;
%for alpha = alpha_values
    %hold on
%    plot(Alpha_history(:,i), '-', 'DisplayName','cos(3x)')
%    legend(num2str(alpha))
%    i = i + 1;
%end
xlabel('Iterations'); % to label X axis
ylabel('Squared Errors'); % to label Y axis
