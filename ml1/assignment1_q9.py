from​ utils.data_handler ​ import​ *
from​ utils.initializers ​ import​ *
from​ forward ​ import​ *
from​ backward ​ import​ *
import​ numpy ​ as​ np
import​ cPickle ​ as​ pkl
import​ json
import​ matplotlib.pyplot ​ as​ plt

# number of iterations each model is trained for
NUM_ITER = ​ 10000
TRAIN_FILE_LINEAR=​ 'train_linear_regression.csv'
TEST_FILE_LINEAR = ​ 'test_linear_regression.csv'
train_lin, test_lin = getTrainTestData(TRAIN_FILE_LINEAR, TEST_FILE_LINEAR)

TRAIN_FILE_LOGISTIC=​ 'train_logistic_regression.csv'
TEST_FILE_LOGISTIC = ​ 'test_logistic_regression.csv'
train_logistic, test_logistic = getTrainTestData(TRAIN_FILE_LOGISTIC,TEST_FILE_LOGISTIC)

# training of linear regression model with squared error
for​ _ ​ in​ range(NUM_ITER):
wts = grad_logistic_regression_err(train_logistic, wts, ​ 0.05​ )

train_err = logistic_regression_err(train_logistic, wts)
test_err = logistic_regression_err(test_logistic, wts)
print​ ​ "Train Error: {} : Test Error: {}"​ .format(train_err, test_err)
# training of linear regression model with absolute error
for​ _ ​ in​ range(NUM_ITER):
wts = grad_abs_err(train_lin, wts, ​ 0.05​ ) ​ # returning squared error as
# well as updating it
# error
train_err = abs_err(train_lin, wts)
test_err = abs_err(test_lin, wts)
print​ ​ "Train Error: {} : Test Error: {}"​ .format(train_err, test_err)
# training of linear regression model with absolute error
for​ _ ​ in​ range(NUM_ITER):
wts = grad_abs_err(train_lin, wts, ​ 0.05​ ) ​ # returning squared error as
# well as updating it
# error
train_err = abs_err(train_lin, wts)
test_err = abs_err(test_lin, wts)
print​ ​ "Train Error: {} : Test Error: {}"​ .format(train_err, test_err)

Import math
# decorator function to calculate the mean error
def​ ​ calc_mean​ (func):
def​ ​ inner​ (data, weights):
mean_loss = 0.
if​ len(data) == ​ 0 ​ :
print​ ​ "Cannot divide!, no input data"
return
mean_loss = ________(fill blank i)________
return​ mean_loss
return​ inner
# function to calculate the φ(x, weights), the output of the regression
model
def​ ​ phi​ (X, weights):
out = ​ 0.
for​ x, wt ​ in​ zip(X, weights):
________(fill blank ii)________
return​ out
# function to calculate the squared error loss
@calc_mean
def​ ​ squared_err​ (data, weights):
err = ​ 0.0
for​ (X, y) ​ in​ data:
________(fill blank iii)________
return​ err / 2.

# function to calculate the absolute error loss
@calc_mean
def​ ​ abs_err​ (data, weights):
err = ​ 0.
for​ (X, y) ​ in​ data:
________(fill blank iv)________
return​ err
def​ ​ sigmoid​ ( ​ X ​ , ​ weights​ ):
sigmoid =
________(fill blank v)________
return​ sigmoid
@calc_mean
def​ ​ logistic_regression_err​ ( ​ data​ , ​ weights​ ):
err = ​ 0.
for​ (X, y) ​ in​ data:
h = ​ ________(fill blank vi)________
err += ​ ________(fill blank vii)________
return​ err