function [theta1, theta2, tacc_history, tracc_history, cost_history, ...
    terrors, trerrors] = trainNN(theta1, theta2, X, y, miniBatchSize, ...
    alpha, epochs, lambda, labels, mu, XTest, labelsTest)
%% Comments %%
% theta1 = Weights and biases for hidden layer (input l x hidden l)
% theta2 = Weights and biases for output layer (hidden l +1 x output l)
% cost = The cost, we use the cross-entropy-cost to prevent learning slowdowns
% X = Training set of 60,000. Matrix of 60,000 x input l
% y = Labels for training set. Matrix of 60,000 x output l
% miniBatchSize = Size of the mini batch used in SGD
% alpha = Learning rate, "size of step" for Stochastic gradient descent
% epochs = Number of iterations to train the neural net for
% lambda = Factor for determining amount of weight decay.
% labels = the raw labels, +1. i.e. if X(i) = 7, label(i) = 8.
% mu = momentum coefficient
% tacc_history = Stores test accuracy to plot in a graph later
% tracc_history = Stores training accuracy to plot in a graph later
% cost_history = Stores costs to plot in graph later
% trerrors = Stores training error to plot learning curve 
% terrors = Stores test error to plot learning curve
%% Initialization and sets theta1 and theta2 %%
m = miniBatchSize;
cost_history = zeros(epochs, 1);
tracc_history = zeros(epochs, 1);
tacc_history = zeros(epochs, 1);
terrors = zeros(epochs,1);
trerrors = zeros(epochs,1);
numBatches = (size(X,1))/m;
v1 = 0;
v2 = 0;
w2 = theta2; 
w1 = theta1;
w2(1,:) = 0; %Sets biases to 0 so unaffected by weight decay
w1(1,:) = 0; %Sets biases to 0 so unaffected by weight decay
%% Feedforward, Backprop, and Mini-Batch SGD %%
    for epoch = 1:epochs
        for i = 1:numBatches
            a1 = [ones(m,1) X(i:(i+(miniBatchSize-1)), :)]; %add column of ones so biases are unaffected by data when matrices are multiplied
            yT = y(i:(i+(miniBatchSize-1)), :);
            
            % Feedforward
            z2 = a1*theta1; % m x hl
            a2 = [ones(m,1) sigmoid(z2)]; %m x hl+1
            z3 = a2*theta2;
            h = sigmoid(z3); % m x o
            reg = (lambda/(2*m))*(sum(sum(w1.^2))+sum(sum(w2.^2)));
            cost = (1/m)*(sum(sum( (-yT.*log(h)) - ((1-yT).*log(1-h)) ))) + reg;
        
            % Backpropagation for weights
            d3 = h - yT; % m x o
            d2 = d3 * (theta2(2:end,:))' .* sigmoidPrime(z2); % m x hl
            % (m x hl) .* (m x hl)
            
            % Gradient Descent
            nablaT2 = (1/m) * (a2'*d3 + lambda*w2); 
            nablaT1 = (1/m) * (a1'*d2 + lambda*w1); 
        	
        	v1 = mu*v1 - alpha*nablaT1;
        	theta1 = theta1 + v1;
            
            v2 = mu*v2 - alpha*nablaT2;
            theta2 = theta2 + v2;
            
            cost_history(epoch) = cost;
        end
        fprintf('\nEpoch: %f\n', epoch);
        
        [correct, errors] = predict(theta1, theta2, X, labels);
        trainAccuracy = correct/size(X,1) *100;
        fprintf('\nTraining Accuracy: %f\n', trainAccuracy);
        tracc_history(epoch) = trainAccuracy;
        errorAcc = errors/size(X,1) *100;
        trerrors(epoch) =  errorAcc;
        
        [correct, errors] = predict(theta1, theta2, XTest, labelsTest);
        testAccuracy = correct/size(XTest,1) *100;
        fprintf('\nTest Accuracy: %f\n', testAccuracy);
        tacc_history(epoch) = testAccuracy;
        errorAcc = errors/size(XTest,1) *100;
        terrors(epoch) =  errorAcc;
    end
end