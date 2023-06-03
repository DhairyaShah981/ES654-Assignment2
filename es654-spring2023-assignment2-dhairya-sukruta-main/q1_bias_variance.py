import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import numpy as np

np.random.seed(1234)
x = np.linspace(0, 10, 50)
eps = np.random.normal(0, 5, 50)
y = x**2 + 1 + eps

# for plotting
# plt.plot(x, y, 'o')
# plt.plot(x, x**2 + 1, 'r-')
# plt.show()

# Plotting bias and variance vs tree depth (increasing complexity)

np.random.seed(1234)
x = np.linspace(0, 10, 50)
eps = np.random.normal(0, 5, 50)
y = x**2 + 1 + eps

depths = range(1, 8)
bias = np.zeros(len(depths))
variance = np.zeros(len(depths))

n_bootstrap = 1000
n_samples = x.shape[0]

bootstrap_samples = np.zeros((n_bootstrap, n_samples))
bootstrap_labels = np.zeros((n_bootstrap, n_samples))

for i in range(n_bootstrap):
    idx = np.random.choice(n_samples, size=n_samples, replace=True)
    bootstrap_samples[i] = x[idx]
    bootstrap_labels[i] = y[idx]



for d, depth in enumerate(depths):
    arr = []
    for i in range(n_bootstrap):
        model = DecisionTreeRegressor(max_depth=depth)
        x_train = bootstrap_samples[i].reshape(-1, 1)
        y_train = bootstrap_labels[i]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_train)
        arr.append(y_pred)
        bias[d] += np.mean((y_train - y_pred)**2)
    for j in range(n_samples):
        lst = []
        for k in range(n_bootstrap):
            lst.append(arr[k][j])
        variance[d] += np.var(np.array(lst))
    variance[d] /= n_samples
    bias[d] /= n_bootstrap


# plot the bias and variance vs depth
plt.plot(depths, bias, label='Bias')
plt.plot(depths, variance, label='Variance')
plt.title('Bias and Variance vs Depth')
plt.xlabel('Depth')
plt.ylabel('Bias/Variance')
plt.legend()
plt.show()
