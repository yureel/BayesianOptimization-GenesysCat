import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, RationalQuadratic, Matern
from scipy.stats import norm
from scipy.optimize import differential_evolution, minimize, rosen
import scipy




# define design space, top row are minima, bottom row maximum values
min_S = [-5, -5, -5, -5, -5]
max_S = [10, 10, 10, 10, 10]

dim = len(min_S)
design_space = np.vstack((np.array(min_S), np.array(max_S)))
initial_data = True
initial_data_file = "BO_log_rosen.csv"

n_init = 50
num_iterations = 950

bounds = []
for i in range(len(min_S)):
    bounds.append((min_S[i], max_S[i]))


if initial_data:
    init_array = pd.read_csv(initial_data_file).values
    n_exp = 0
    for i in range(init_array.shape[0]):
        if np.isnan(init_array[i, 0]):
            n_exp = i
            break
        n_exp = i
    x_init = init_array[:n_exp, :dim]
    y_init = init_array[:n_exp, -1]
else:
    # range of x values
    x_init = np.add(np.random.rand(n_init, dim)*(design_space[1, :] - design_space[0, :]), design_space[0, :])
    y_init = np.empty((n_init, 1))
    # y_init = run_H_S_model(x_init)
    for i in range(x_init.shape[0]):
        print("initial experiment ",i)
        y_init[i, 0] = -1*rosen(x_init[i, :])
        print(y_init[i, 0])
    # remove all failed simulations from initial experiments
    print(np.where(y_init == -10))
    x_init = np.delete(x_init, np.where(y_init == -10), 0)
    y_init = np.delete(y_init, np.where(y_init == -10), 0)
    df = pd.DataFrame(np.hstack((x_init, y_init)))
    df.to_csv(initial_data_file, mode='a', header=False, index=False)


def expected_improvement(x, gp_model, best_y, gp_model_std=None):
    # calculate EI for points x, given a model and current maximum
    y_pred, y_std = gp_model.predict(x, return_std=True)
    a = y_pred-best_y
    z = np.divide(a, y_std.reshape(-1, 1))
    ei = np.multiply(a, norm.cdf(z)) + np.multiply(y_std.reshape(-1, 1), norm.pdf(z))
    # z = (y_pred - best_y) / y_std
    # ei = (y_pred - best_y) * norm.cdf(z) + y_std * norm.pdf(z)
    return ei


def eval_ei(x):
    y_pred, y_std = gp_model.predict(x.reshape(1, -1), return_std=True)
    y_useless, y_std = gp_model_std.predict(x.reshape(1, -1), return_std=True)
    a = y_pred-best_y
    z = np.divide(a, y_std.reshape(-1, 1))
    ei = np.multiply(a, norm.cdf(z)) + np.multiply(y_std.reshape(-1, 1), norm.pdf(z))
    # z = (y_pred - best_y) / y_std
    # ei = (y_pred - best_y) * norm.cdf(z) + y_std * norm.pdf(z)
    return -1*ei


def log_ei(x, gp_model, best_y):
    y_pred, y_std = gp_model.predict(x, return_std=True)
    a = y_pred-best_y
    z = np.divide(a, y_std.reshape(-1, 1))
    eps = 1e-6
    c1 = np.log10(2*np.pi)/2
    c2 = np.log10(np.pi/2)/2
    if z > -1:
        log_h = np.log10(norm.pdf(z)+np.multiply(z, norm.cdf(z)))
    elif -1/np.sqrt(eps) < z <= -1:
        d = -z/np.sqrt(2)
        b = c2 + np.log10(np.exp(d**2)*scipy.special.erfc(d)*abs(z))
        log_h = -z**2/2-c1+np.log10(1-np.exp(b))
    else:
        log_h = -z**2/2-c1-2*np.log10(np.abs(z))
    if y_std < 1e-10:
        return log_h - 10
    return log_h + np.log10(y_std)


def eval_log_ei(x):
    return -1*log_ei(x.reshape(1, -1), gp_model, best_y)


def gp_prediction(x):
    return -1*gp_model.predict(x.reshape(1, -1))


kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
kernel_2 = Matern(length_scale=np.ones(dim), nu=1.5, length_scale_bounds=(1e-10, 1e10))
kernel_1 = RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-10, 1e10))

# gp_model = GaussianProcessRegressor(kernel=kernel_1, alpha=1e-5)
gp_model = GaussianProcessRegressor(kernel=kernel+kernel_1+kernel_2, alpha=1e-5)
# Fit the Gaussian process model to the sampled points
gp_model.fit(x_init, y_init)
gp_model_std = GaussianProcessRegressor(kernel=kernel)
gp_model_std.fit(x_init, y_init)

# Generate predictions using the Gaussian process model
y_pred, y_std = gp_model.predict(x_init, return_std=True)

x_samples = x_init
y_samples = y_init.reshape(-1, 1)

for i in range(num_iterations):
    print("iteration ",i)
    # Fit the Gaussian process model to the sampled points
    gp_model.fit(x_samples, y_samples)
    gp_model_std.fit(x_samples, y_samples)

    # Determine the point with the highest observed function value
    best_idx = np.argmax(y_samples)
    best_x = x_samples[best_idx]
    best_y = y_samples[best_idx]


    if i % 2 == 0:
        # every two runs exploit the surrogate model

        potential_x = []
        potential_y = []
        from scipy.optimize import minimize
        res = minimize(gp_prediction, best_x, bounds=bounds)
        potential_x.append(res.x)
        potential_y.append(gp_model.predict(res.x.reshape(1, -1))[0])
        pool = np.add(np.random.rand(10, dim) * (design_space[1, :] - design_space[0, :]),
                      design_space[0, :])
        for j in range(10):
            res = minimize(gp_prediction, pool[j, :], bounds=bounds)
            potential_x.append(res.x)
            potential_y.append(gp_model.predict(res.x.reshape(1, -1))[0])
        res = differential_evolution(gp_prediction, bounds)
        new_x = res.x
    else:
        potential_x = []
        potential_y = []
        res = minimize(eval_ei, best_x+0.1, bounds=bounds)
        potential_x.append(res.x)
        potential_y.append(-1*eval_ei(res.x))
        pool = np.add(np.random.rand(10, dim) * (design_space[1, :] - design_space[0, :]),
                      design_space[0, :])
        for j in range(10):
            res = minimize(eval_ei, pool[j, :], bounds=bounds)
            potential_x.append(res.x)
            potential_y.append(-1*eval_ei(res.x))
        res = differential_evolution(eval_ei, bounds)
        potential_x.append(res.x)
        potential_y.append(-1*eval_ei(res.x))
        new_x = potential_x[np.argmax(potential_y)]

    print(new_x)
    new_y = -1*rosen(new_x)
    if isinstance(new_y, np.float64) or isinstance(new_y, float):
        new_y = np.array([new_y])
    if new_y[0] == -10:         # error recovery value
        if gp_model.predict(new_x.reshape(1, -1))[0] > best_y:
            new_y = 10*best_y       # the recovered value cannot be higher than the current best_y otherwise it diverges
        else:
            new_y = gp_model.predict(new_x.reshape(1, -1))[0]     # set recovery value to model prediction
        print("recovered: ", new_y)
    predicted_new_y = gp_model.predict(new_x.reshape(1, -1))[0]
    print("measured value: ", new_y)
    print("model value: ", predicted_new_y)
    print("model discrepancy: ",abs(new_y-predicted_new_y))
    x_samples = np.vstack((x_samples, new_x))
    y_samples = np.vstack((y_samples, new_y))
    print(new_x, new_y)
    df = pd.DataFrame(np.concatenate((new_x, new_y)).reshape(1, -1))
    df.to_csv(initial_data_file, mode='a', header=False, index=False)

best_idx = np.argmax(y_samples)
best_x = x_samples[best_idx]
best_y = y_samples[best_idx]

print(y_samples)
print(best_y, best_x)

