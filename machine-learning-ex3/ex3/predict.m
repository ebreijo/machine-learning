function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

z_j = Theta1 * X'; % compute z of the second layer (hidden layer)
assignin('base','z_j',z_j);
a_j = sigmoid(z_j); % compute the activation units of the hidden layer

a_j = [ones(1, m); a_j]; % Add bias unit to the activation units of the hidden layer
assignin('base','a_j',a_j);

z_j_next = Theta2 * a_j; % Compute z of the output layer
assignin('base','z_j_next',z_j_next);
h_x = sigmoid(z_j_next); % compute the output (predicted result)
assignin('base','h_x',h_x);

[M, I] = max(h_x); % Get the max and index of each column
assignin('base','M',M);
assignin('base','I',I);
p = I'; % The index where the max was found is the predicted number



% =========================================================================


end
