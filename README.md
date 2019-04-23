# Function Approximation with Neural Networks
This short project practically demonstrates the ability of neural networks to approximate simple functions.

I have studied the Universal Approximation Theorem (UAT) for Neural Networks (see https://en.wikipedia.org/wiki/Universal_approximation_theorem) but was curious to validate this practically for simple functions. Specifically, I approximated the following functions with simple NNs:
* Identity function: f(x) = x
* Simple linear function: f(x) = x * 1/3 - 6
* Non-linear polynomial function: f(x) = x ** 1/3 - 10 

Clearly it is not possible to use an infinitely wide hidden layers (as specified by the UAT to get an exact approximation), so I manually chose network designs which were able to approximate the above to a required tolerance.

### Degeneracy
This project also demonstrated the degeneracy of NN weight solutions when there are more degrees of freedom than are required to solve the problem. E.g. In the case of the identity function, there were multiple (in fact infinitely many) degenerate solutions that gave the same outputs (with a non-linear hidden layer).

### To run
Clone the repository
run the following from the home repository
```pip install -r requirements.txt
python nn_function_approximator.py```

Use python 3.X
