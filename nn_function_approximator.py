# -*- coding: utf-8 -*-
"""NN_function_approximator.py

Short experiment to validate that a NN is able to learn
 functions easily (i.e. to confirm this "universal
function approximator" idea)

"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


#Fix all seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)


class VanillaNN(nn.Module):
    def __init__(self, input_size, output_size, num_hidden, hidden):
        super(VanillaNN, self).__init__()
        assert num_hidden == len(hidden)
        if not hidden:
            self.fclayers = nn.ModuleList([nn.Linear(input_size, output_size)])
        else:
            fc1 = nn.Linear(input_size, hidden[0])
            self.fclayers = nn.ModuleList([fc1])
        for layer_idx in range(num_hidden - 1):
            fc = nn.Linear(hidden[layer_idx], hidden[layer_idx + 1])
            self.fclayers.append(fc)
        fc_last = nn.Linear(hidden[-1], output_size)
        self.fclayers.append(fc_last)

        #initialize
        for fc in self.fclayers:
            nn.init.xavier_uniform_(fc.weight)

        self.lReLU = nn.LeakyReLU(negative_slope=0.05, inplace=False)

    def forward(self, x):

        for fc in self.fclayers[:-1]:
            x = self.lReLU(fc(x))

        x = self.fclayers[-1](x)

        return x

#     def parameters(self):
#         return [x.weight for x in self.fclayers]

#generate the test and training set from a uniform distribution

def gen_test_set(test_size, maximum):
    assert test_size < maximum
    test = torch.randint(0, maximum, (test_size,)).type(torch.FloatTensor)
    test = test.reshape((-1, 1))
    return test

def gen_number_not_in_test(test, maximum):
    while True:
        value = np.random.randint(0, maximum)
        while value in test:
            value = np.random.randint(0, maximum)
        yield value

def gen_train_batch(test, maximum, batch_size):

    matrix = np.fromiter(gen_number_not_in_test(test, maximum), int, count=batch_size)
    matrix = torch.tensor(matrix).type(torch.FloatTensor)
    matrix = matrix.reshape((-1, 1))
    return matrix

def approximate_function(model, fn, name, fn_mean, fn_std):
    """Takes a model and a function and trains.
    Must pass the valid function mean and std to successfully normalize """

    print("Attempting to approximate {} with the following model:".format(name))
    print(model)

    optimizer = optim.Adam(model.parameters(), learning_rate)
    loss_function = nn.MSELoss()

    #Define local helper functions to normalize using the mean and std
    #using the mean and std

    def normalize_x(x):
        return (x - mean) / std

    def unnormalize_y(y):
        return y * fn_std + fn_mean

    print("Inputs: mean = {:.2f}, std = {:.2f}".format(mean, std))
    print("Function: mean = {:.2f}, std = {:.2f}".format(fn_mean, fn_std))

    #generate test set and normalize
    test = gen_test_set(test_set_size, max_size)
    test_norm = normalize_x(test)

    #generate test labels
    y_test = test.apply_(fn)

    #training loop:
    for batch in range(total_batches):
        # During training, I will skip over all values in
        # the test_set so they remain unseen
        train_batch = gen_train_batch(test, max_size, train_batch_size)
        X = normalize_x(train_batch)


        model.train()  # put model to training mode
        optimizer.zero_grad()

        preds = unnormalize_y(model(X))
        y = train_batch.apply_(fn)

        loss = loss_function(preds, y)

        loss.backward()

        optimizer.step()

        #print('Batch %d, loss = %.4f' % (batch, loss.item()))
        if batch % eval_every == 0:

            model.eval()
            pred = model(test_norm)

            pred = unnormalize_y(pred)
            loss_test = loss_function(pred, y_test).item()

            print('Batch %d, loss = %.4f, loss_test = %.4f' % (batch, loss.item(), loss_test))


    print()
    print("Printing weights...")
    for fc in model.fclayers:
        print(fc.weight)

    model.eval()
    pred = model(test_norm)

    pred = unnormalize_y(pred)
    loss_test = loss_function(pred, y_test).item()

    print("Printing results and sample outputs...")
    print('loss = %.4f, loss_test = %.4f' % ( loss.item(), loss_test))
    print(pred[0:4])
    print(y_test[0:4])

    #Now test results. Just to 3.s.f as we are not waiting for convergence
    assert torch.allclose(pred, y_test, rtol=1e-02)

    print("ASSERT PASSED. FUNCTION: \'{}\' SUCCESSFULLY APPROXIMATED".format(name.upper()))
    print()

def sample(fn, n_samples = 10000):
    """Returns sampled mean and std of function"""
    inputs = gen_train_batch([], n_samples, train_batch_size)
    x = inputs.apply_(fn).numpy()
    mean = np.mean(x)
    np.sqrt(np.mean(abs(x - mean)**2))
    return mean, std


def check_identity():
    """Validate that a network with a single hidden neuron
    (and positive inputs) can learn the identity mapping"""

    #single hidden layer with 1 hidden units.
    #It should be possible to fully approximate with this
    num_hidden = 1
    hidden = [1]

    model = VanillaNN(1, 1, num_hidden, hidden)

    ## DEFINE FUNCTION TO OPTIMIZE - check that it can learn the identity
    def identity(x):
        return float(x)

    #For uniform distribution the mean and std are as follows:
    fn_mean = mean
    fn_std = std


    approximate_function(model, identity, "Identity", fn_mean, fn_std)

def check_linear_poly():

    num_hidden = 1
    hidden = [2]

    model = VanillaNN(1, 1, num_hidden, hidden)

    ## DEFINE FUNCTION TO OPTIMIZE - check that it can learn the identity
    def poly(x):
        return float(x * 1/3 - 6)

    #For uniform distribution the mean and std are as follows:
    fn_mean = poly(max_size) / 2.0
    fn_std = poly(max_size) / np.sqrt(12) #When function is linear in x

    approximate_function(model, poly, "Linear polynomial", fn_mean, fn_std)


def check_non_lin_poly():

    # Two layers of non-linearity to
    # efficiently approximate the non-linear polynomial
    # It should also be possible to use a single hidden layer
    # and more hidden neurons
    num_hidden = 2
    hidden = [4, 4]

    model = VanillaNN(1, 1, num_hidden, hidden)

    ## DEFINE FUNCTION TO OPTIMIZE - check that it can learn the identity
    def poly(x):
        return float(x ** 1/3 - 10)

    #expectation calculated by hand
    # e.g. see: https://en.wikipedia.org/wiki/Variance#Continuous_random_variable

    # fn_mean = max_size ** 5 / 5.0 - 10.0 * max_size ** 2 / 2.0
    # fn_expectation_x_2 = max_size ** 6 / 6.0 - 10.0 * max_size ** 3 / 3.0
    # fn_std = np.sqrt(fn_expectation_x_2 - fn_mean ** 2)

    #expectation and variance sampled
    fn_mean, fn_std = sample(poly)
    approximate_function(model, poly, "non-linear polynomial", fn_mean, fn_std)



if __name__ == "__main__":

    #hyperparameters - I will use these as
    #(effective) global variables within main()

    test_set_size = 100
    max_size = int(1e4)
    train_batch_size = test_set_size * 2
    total_batches = 1000
    learning_rate = 0.05
    eval_every = 50



    #For uniform distribution the mean and std are as follows:
    mean = float(max_size / 2)
    std = float(max_size / np.sqrt(12))

    #Check three type of function
    check_identity()
    check_linear_poly()

    #Change hyperparameters for non-linear case
    learning_rate = 0.01
    total_batches = 1200
    check_non_lin_poly()
