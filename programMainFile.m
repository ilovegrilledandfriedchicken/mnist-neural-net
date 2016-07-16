%% Initialization %%
clear ; close all; clc;

%% Loads Data %%
x = loadMNISTImages('train-images-idx3-ubyte');
X = x';
labels = loadMNISTLabels('train-labels-idx1-ubyte');
labels = labels+1;

temp = eye(10);
y = temp(labels,:); %Makes y a 60,000 x 10 matrix, where each column  
                      %corresponds to 0s and 1 one more than the position  
                      %of its number, so a label of 7 = [0 0 0 0 0 0 0 1 0 0]


XTest = loadMNISTImages('t10k-images-idx3-ubyte');
XTest = XTest';
labelsTest = loadMNISTLabels('t10k-labels-idx1-ubyte');
yTest = temp(labelsTest+1,:);

fprintf('Data has loaded, press enter to continue.\n');
pause;
%% Set Neural Net Structure %%
input = size(X,2);% Size of input layer (784)
hidden = 1000; % Size of hidden layer
output = size(y,2); % Size of output layer (10)
m = size(X,1);


stdw = 1/sqrt(input);                          % Initializes weights and  
mean = 0;                                      % biases using normal  
weights1 = stdw.*randn(hidden, input) + mean;  % distribution with set  
weights2 = stdw.*randn(output, hidden) + mean; % standard deviation and 
stdb = 1;                                      % mean to prevent learning
bias1 = stdb.*randn(hidden,1) + mean;          % slowdown
bias2 = stdb.*randn(output,1) + mean;
theta1 = [bias1 weights1]'; %Adds row of biases to param matrix, so 1st row
theta2 = [bias2 weights2]'; %is biases, followed by rows of all the weights
%% Set Hyper-parameters %%
miniBatchSize = 500; %size of mini batch used
epochs = 50; %iterations to train for
lambda = 0; %regularization/weight decay parameter
alpha = 0.01; %learning rate
mu = 0.9; %momentum coefficient

fprintf('Press enter to start training.\n');
pause;
fprintf('Press wait while training completes.\n');

%% Start Training (Comment out if softmax)%%

[theta1, theta2, tacc_history, tracc_history, cost_history, ...
    terrors, trerrors] = trainNN(theta1, theta2, X, y, miniBatchSize, ...
    alpha, epochs, lambda, labels, mu, XTest, labelsTest);

fprintf('Training finished, press enter to see convergence graph.\n');
pause;
%% Plot Convergence Graph
figure;
plot(1:numel(cost_history), cost_history, '-b', 'LineWidth', 2);
xlabel('Epoch');
ylabel('Cost');

fprintf('Press enter to display training accuracy graph.\n');
pause;

%% Plot Training Accuracy Graph %%
figure;
plot(1:numel(tracc_history), tracc_history, '-b', 'LineWidth', 2);
xlabel('Epoch');
ylabel('Training Accuracy');

fprintf('Press enter to display test accuracy graph.\n');
pause;

%% Plot Test Accuracy Graph %%
figure;
plot(1:numel(tacc_history), tacc_history, '-b', 'LineWidth', 2);
xlabel('Epoch');
ylabel('Test Accuracy');

fprintf('Press enter to display learning curve.\n');
pause;

%% Plot Learning Curve
figure;
plot(1:numel(trerrors), trerrors, 1:numel(terrors), terrors);
legend('Train', 'Test')
xlabel('Epoch');
ylabel('Accuracy');

fprintf('Press enter to display test accuracy.\n');
pause;