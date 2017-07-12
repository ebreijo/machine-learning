function error_test = testCurve(theta, Xtest, ytest)%TESTCURVE Generates the test set errors needed %to plot a test curve%   error_test = ...%       TESTGCURVE(theta, Xtest, ytest) returns the test error %% You need to return this values correctlyerror_test = 0;    % Compute the test error using the test set with the trained theta% Note that the test error does not include the regularization term.% So lambda should be 0.error_test = linearRegCostFunction(Xtest, ytest, theta, 0);  % -------------------------------------------------------------% =========================================================================end