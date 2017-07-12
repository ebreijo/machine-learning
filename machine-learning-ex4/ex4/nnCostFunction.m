function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Feedforward propagation to find the predicted result h(x)
z_hidden_layer = Theta1 * X'; % compute z of the second layer (hidden layer)
assignin('base','z_hidden_layer',z_hidden_layer);
a_hidden_layer = sigmoid(z_hidden_layer); % compute the activation units of the hidden layer

a_hidden_layer = [ones(1, m); a_hidden_layer]; % Add bias unit to the activation units of the hidden layer
assignin('base','a_hidden_layer',a_hidden_layer);

z_output_layer = Theta2 * a_hidden_layer; % Compute z of the output layer
assignin('base','z_output_layer',z_output_layer);
h_x = sigmoid(z_output_layer); % compute the output (predicted result)
assignin('base','h_x',h_x);

% Recode the labels as vectors containing only values 0 or 1
y_recoded = zeros(num_labels, m); 
for i=1:m,
  y_recoded(y(i),i) = 1;
end
assignin('base','y_recoded',y_recoded);


% Compute cost function J with regularization
unregularized_cost = -1/m * sum(sum(y_recoded .* log(h_x)) + sum((1 - y_recoded) .* log(1 - h_x)));
assignin('base','unregularized_cost',unregularized_cost);
regularization = lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
assignin('base','regularization',regularization);
J = unregularized_cost + regularization;


% Backpropagation Algoeithm 

for t=1:m

% Set a_1 to x(t)
a_1 = X(t,:);
assignin('base','a_1',a_1);

% Forward propagation to find predicted result
z_2 = Theta1 * a_1'; % compute z of the second layer (hidden layer)
assignin('base','z_2',z_2);
a_2 = sigmoid(z_2); % compute the activation units of the hidden layer

a_2 = [1; a_2]; % Add bias unit to the activation units of the hidden layer
assignin('base','a_2',a_2);

z_3 = Theta2 * a_2; % Compute z of the output layer
assignin('base','z_3',z_3);
a_3 = sigmoid(z_3); % compute the output (predicted result)
assignin('base','a_3',a_3);

% Compute the error values delta
delta_3 = a_3 - y_recoded(:,t); % Compute the error values for the output layer
assignin('base','delta_3',delta_3);
sigmoidGrad = [1; sigmoidGradient(z_2)];
delta_2 = Theta2' * delta_3 .* sigmoidGrad; % Compute the error values for the hidden layer
assignin('base','delta_2',delta_2);

% Compute de gradient
Theta2_grad = Theta2_grad + delta_3 * (a_2'); % Accumulate the gradient Theta2
assignin('base','Theta2_grad',Theta2_grad);
delta_2 = delta_2(2:end);
Theta1_grad = Theta1_grad + delta_2 * (a_1); % Accumulate the gradient Theta1
assignin('base','Theta1_grad',Theta1_grad);

end


% Obtain the regularized gradient for the neural network
Theta1_grad(:,1) = (1/m) .* Theta1_grad(:,1);
Theta1_grad(:,2:end) = (1/m) .* Theta1_grad(:,2:end) + (lambda/m) .* Theta1(:,2:end);
Theta2_grad(:,1) = (1/m) .* Theta2_grad(:,1);
Theta2_grad(:,2:end) = (1/m) .* Theta2_grad(:,2:end) + (lambda/m) .* Theta2(:,2:end);




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
