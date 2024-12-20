import numpy as np

# Gradient Descent Optimizer
class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update_parameters(self, params, grads):
        updated_params = []
        for param, grad in zip(params, grads):
            updated_param = param - self.learning_rate * grad
            updated_params.append(updated_param)
        return updated_params

# Adam Optimizer
class AdamOptimizer:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update_parameters(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]

        self.t += 1
        learning_rate_t = self.learning_rate * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)

        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            updated_param = param - learning_rate_t * m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_params.append(updated_param)

        return updated_params

# Test Function and Gradient

def quadratic_function(x):
    return np.sum(x**2)

def quadratic_gradient(x):
    return 2 * x

# Testing Optimizers
params = [np.array([3.0, 4.0])]
gd_optimizer = GradientDescentOptimizer(learning_rate=0.1)
adam_optimizer = AdamOptimizer(learning_rate=0.1)

num_iterations = 100
for i in range(num_iterations):
    grads = [quadratic_gradient(param) for param in params]
    params = gd_optimizer.update_parameters(params, grads)
    params = adam_optimizer.update_parameters(params, grads)

    if (i + 1) % 10 == 0:
        print(f"Iteration {i+1}: Quadratic function value = {quadratic_function(params[0])}")

print("Final parameters:")
print(params)

# Comparison with Keras Built-in Optimizers (to be implemented)
