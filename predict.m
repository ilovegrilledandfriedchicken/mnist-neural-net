function [correct, errors] = predict(theta1, theta2, X, labels)
%% Initialize some parameters %%
m = size(X, 1);
predictions = zeros(size(X, 1), 1);
%% Code %%
X = [ones(m,1) X]; % m x il+1
z2 = X*theta1; % m x hl
a2 = [ones(m,1) sigmoid(z2)]; %m x hl+1
z3 = a2*theta2; % m x o
hypothesis = sigmoid(z3); %m x o
[~, predictions] = max(hypothesis, [], 2); %Returns vector of indices of NN's guess
correct = sum(double(predictions == labels));
errors = m - correct; 
end
