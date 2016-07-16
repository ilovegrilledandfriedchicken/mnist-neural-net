function g = sigmoidPrime(z)
g = zeros(size(z));
g = sigmoid(z) .* (1-sigmoid(z));
end