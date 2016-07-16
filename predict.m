function [correct, errors] = predict(theta1, theta2, X, labels)
%% Initialize some parameters %%
m = size(X, 1);
predictions = zeros(size(X, 1), 1);
correct = 0;
errors = 0;
%% Code %%
X = [ones(m,1) X];
z2 = X*theta1;
a2 = [ones(m,1) sigmoid(z2)]; %m x 101
z3 = a2*theta2;
hypothesis = sigmoid(z3); %m x 10
[~, predictions] = max(hypothesis, [], 2); %Returns vector of indices of NN's guess
correct = sum(double(predictions == labels));
errors = m - correct; 
end
