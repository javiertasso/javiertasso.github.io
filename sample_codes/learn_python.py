import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import GMM
from scipy import stats
from scipy.stats import norm
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
import autograd.numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)



# -------------------------------------------------------
# Estimate a linear model -------------------------------
# -------------------------------------------------------

# 0 - Simulate some data --------------------------------
n = 100
X = np.random.multivariate_normal([50, 50], [[25, 10], [10, 25]], n)
X = np.hstack((np.ones((n, 1)), X))
epsilon = np.random.normal(0, 1, n)
y = X @ np.array([25, -10, 10]) + 40*epsilon

# 1 - Using commands ------------------------------------
linear_model = sm.OLS(y, X).fit()
print(linear_model.summary())

# 2 - Using Linear Algebra ------------------------------
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
sigma_hat_sq = np.sum((y - X @ beta_hat)**2) / (n - 3)
beta_hat_var = sigma_hat_sq * np.linalg.inv(X.T @ X)
std_err = np.sqrt(np.diag(beta_hat_var))
t_stats = beta_hat / std_err
p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - 3))
print(f"Using Linear Algebra:")
print(f"Estimated coefficients: {beta_hat}")
print(f"Standard errors: {std_err}")

# 3 - Using FWL Theorem (only for beta_1) ---------------
y_dem = y - np.mean(y)
x1_dem = X[:, 1] - np.mean(X[:, 1])
x2_dem = X[:, 2] - np.mean(X[:, 2])
beta_aux_12 = 1/(x2_dem.T @ x2_dem) * (x2_dem.T @ x1_dem)
residuals_12 = x1_dem - beta_aux_12 * x2_dem
beta_aux_y2 = 1/(x2_dem.T @ x2_dem) * (x2_dem.T @ y_dem)
residuals_y2 = y_dem - beta_aux_y2 * x2_dem
beta_hat_fwl = 1/(residuals_12.T @ residuals_12) * (residuals_12.T @ residuals_y2)
sigma_hat_sq_fwl = np.sum((residuals_y2 - residuals_12 * beta_hat_fwl)**2) / (n - 3) # Careful: Adjust degrees of freedom 
std_err_fwl = np.sqrt(sigma_hat_sq_fwl / (residuals_12.T @ residuals_12))
print(f"Using FWL Theorem:")
print(f"Estimated coefficient: {beta_hat_fwl}")
print(f"Standard error: {std_err_fwl}")

# -------------------------------------------------------
# Maximum Likelihood Estimation -------------------------
# -------------------------------------------------------

# 0 - Simulate some data --------------------------------
n = 10000
X = np.random.multivariate_normal([0, 0], [[1, 0.2], [0.2, 1]], n)
X = np.hstack((np.ones((n, 1)), X))
beta = np.array([0.5, -0.2, 0.2])
probabilities = norm.cdf(X @ beta)
y = np.random.binomial(1, probabilities)

# 1 - Using commands ------------------------------------
probit_model = sm.Probit(y, X).fit(maxiter=100)
print(probit_model.summary())

# 2 - Using Maximum Likelihood Estimation ---------------
def probit_log_likelihood(beta, X, y):
    probabilities = norm.cdf(X @ beta)
    log_likelihood = np.sum(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))
    return -log_likelihood

def probit_gradient(beta, X, y):
    linear_pred = np.dot(X, beta)
    probs = norm.cdf(linear_pred)
    pdf_values = norm.pdf(linear_pred)
    gradient = np.dot(X.T, (y - probs) * pdf_values)
    return -gradient

def probit_hessian(beta, X, y):
    linear_pred = np.dot(X, beta)
    probs = norm.cdf(linear_pred)
    pdf_values = norm.pdf(linear_pred)
    
    # Diagonal elements for Hessian scaling
    scaling_factors = (pdf_values ** 2) / (probs * (1 - probs))
    
    # Form Hessian
    hessian_matrix = -np.dot(X.T * scaling_factors, X)
    return hessian_matrix

result = minimize(probit_log_likelihood, [0, 0, 0], args=(X, y), 
                  method='Newton-CG', jac=probit_gradient, 
                  hess = probit_hessian,
                  options={'gtol': 1e-8, 'maxiter': 1000})
print("Estimated coefficients:", result.x) 
print("Final log-likelihood:", -result.fun)
hessian = probit_hessian(result.x, X, y)
standard_errors = np.sqrt(np.diag(np.linalg.inv(-hessian)))
print("Standard errors:", standard_errors)


# -------------------------------------------------------
# Generalized Method of Moments Estimation --------------
# -------------------------------------------------------

# 0 - Simulate some data --------------------------------
n = 1000
z1 = np.random.normal(size=n)
x2 = np.random.normal(size=n)
eps = np.random.normal(size=n)
x1 = 1 + 0.5 * z1 + 0.5 * eps 
y = 1 + 0.5 * x1 - 0.3 * x2 + eps
X = np.column_stack([np.ones(n), x1, x2]) 
Z = np.column_stack([np.ones(n), z1, x2])  

# 1 - Using commands ------------------------------------
class IVGMM(GMM):
    def momcond(self, params):
        beta0, beta1, beta2 = params
        residuals = y - (beta0 + beta1 * x1 + beta2 * x2)
        moments = np.column_stack([residuals, residuals * z1, residuals * x2])
        return moments

gmm_model = IVGMM(endog=y, exog=X, instrument=Z)
gmm_result = gmm_model.fit(start_params=[0, 0, 0], maxiter=2, optim_method='bfgs')
print(gmm_result.summary())

# 2 - Using GMM -----------------------------------------
def moment_conditions(beta, y, X, Z):
    residuals = y - X @ beta
    return np.column_stack([residuals, residuals * Z[:, 1], residuals * Z[:, 2]])

def gmm_objective(beta, y, X, Z, W):
    moments = moment_conditions(beta, y, X, Z)
    avg_moment = moments.mean(axis=0)
    return avg_moment.T @ W @ avg_moment

def jacobian(X, Z):
    J = np.zeros((Z.shape[1], X.shape[1]))
    for j in range(X.shape[1]):
        J[:, j] = -np.mean(Z * X[:, [j]], axis=0)
    return J

# First Stage
result_stage1 = minimize(gmm_objective, np.array([0, 0, 0]), args=(y, X, Z, np.eye(3)), method='BFGS')
moments_stage1 = moment_conditions(result_stage1.x, y, X, Z)
optimal_W = np.linalg.inv(moments_stage1.T @ moments_stage1 / n)

# Second Stage 
result_stage2 = minimize(gmm_objective, result_stage1.x, args=(y, X, Z, optimal_W), method='BFGS')
print("GMM estimates:", result_stage2.x)

# Standard Erros 
G = jacobian(X, Z)
var_cov_matrix = np.linalg.inv(G.T @ optimal_W @ G) / n
print("Standard errors:", np.sqrt(np.diag(var_cov_matrix)))

# -------------------------------------------------------
# Kernel Density Estimation -----------------------------
# -------------------------------------------------------

# 0 - Simulate some data --------------------------------
n_samples = 1000
data = np.concatenate([
    np.random.normal(loc=-2, scale=0.8, size=n_samples // 2), 
    np.random.normal(loc=3, scale=1.2, size=n_samples // 2)    
])

plt.hist(data, bins=30, density=True, alpha=0.5, color='skyblue', edgecolor='black')
plt.title("Histogram of Generated Data")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

# 1 - Using commands ------------------------------------
kde = gaussian_kde(data, bw_method=0.25)  
x_vals = np.linspace(data.min() - 1, data.max() + 1, 100)
density_vals = kde(x_vals)

plt.plot(x_vals, density_vals, color='darkblue', lw=2, label='KDE')
plt.hist(data, bins=30, density=True, alpha=0.3, color='skyblue', edgecolor='black')
plt.title("Kernel Density Estimation")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

# 2 - Manually ------------------------------------------  
h = np.sqrt(0.25)
n = len(data)
density_vals_manual = np.zeros_like(x_vals)

for i, x in enumerate(x_vals):
    density_vals_manual[i] = (1 / (h*n)) * np.sum(norm.pdf((x - data) / h))

plt.plot(x_vals, density_vals_manual, color='darkblue', lw=2, label='Manual KDE')
plt.hist(data, bins=30, density=True, alpha=0.3, color='skyblue', edgecolor='black')
plt.title("Manual Kernel Density Estimation")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()


# -------------------------------------------------------
# Particle Filter ---------------------------------------
# -------------------------------------------------------

# 0 - Simulate some data --------------------------------

# Simple model 
# y_t = \psi * s_t + u_t
# s_t = \phi * s_{t-1} + e_t 
# Where u_t and e_t are Normal errors 
# A sample here is a time series {y_t}_{t=1}^T

pphi = 0.5
ppsi = 0.75
sigma_e = 1.00
sigma_u = 0.25

n = 100
s = np.zeros(n)  # State vector
y = np.zeros(n)  # Observation vector
e = sigma_e * np.random.randn(n)  # Innovation terms for the state
u = sigma_u * np.random.randn(n)  # Innovation terms for the observation

s[0] = e[0]  
y[0] = ppsi * s[0] + u[0]

for tt in range(1, n):  # Start from t=1 (second time step)
    s[tt] = pphi * s[tt - 1] + e[tt]
    y[tt] = ppsi * s[tt] + u[tt]

# 1 - Use first 80 observations to estimate parameters ---
data_y = y[:80]

def kalman(params, data_y): 
    pphi, ppsi = params
    sigma_u = 0.25
    sigma_e = 1.00
    T = len(data_y)
    s = np.zeros(T+1)
    P = np.zeros(T+1)  
    pr = np.zeros(T) 

    # Initial Values
    s[0] = 0 
    P[0] = 0.10

    for t in range(0, T):
        # Update s and p 
        s_val = pphi * s[t]
        P_val = pphi**2 * P[t] + sigma_e**2
        F_val = ppsi**2 * P_val + sigma_u**2 
        y_for = ppsi * s_val

        # s_{t} | Y_{1:t}
        s[t+1]  = s_val + P_val * ppsi / F_val * (data_y[t] - y_for)
        P[t+1]  = P_val - P_val**2 * ppsi**2 / F_val
        pr[t] = np.log(norm.pdf(data_y[t], loc=y_for, scale=np.sqrt(F_val)))

    return -np.sum(pr)

params_init = [0.1, 0.1]
result = minimize(kalman, params_init, args=(data_y,), method='BFGS')
print(result.x)
pphi_hat, ppsi_hat = result.x

# 2 - Particle Filter -----------------------------------
    # Given those estimates of pphi_hat and pssi_hat
    # Use a particle filter to filter the remaining 20 obs 
    # Particle filter is an overkill in this linear guassian case
    # But it is still a good example to show how to implement it
filter_y = y[80:]
T = len(filter_y)
true_s = s[80:]

# Set up the particle filter
import numpy as np

def resample(particles, weights):
    # Ensure that weights are non-negative and replace zero or negative weights with a small positive number
    weights = np.maximum(weights, 1e-10)  # Avoid zero weights (replace with small positive value)
    
    # Normalize the weights so that they sum to 1
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        print("Warning: Weight sum is zero, using uniform distribution")
        weights = np.ones(len(particles)) / len(particles)  # If all weights are zero, use uniform distribution
    else:
        weights /= weight_sum  # Normalize to ensure the sum of weights is 1
    
    # Resample using numpy's random.choice with the normalized weights
    indices = np.random.choice(len(particles), size=len(particles), p=weights)
    resampled_particles = particles[indices]
    
    return resampled_particles

def particle(N, filter_y, pphi_hat, ppsi_hat): 
    T = len(filter_y)
    s_curr = np.zeros(N)
    y_curr = np.zeros(N)
    weights = np.zeros(N)
    state_nores = np.zeros((N,T))
    filtered_state = np.zeros((N,T))
    N_eff = np.zeros(T)
    sigma_e = 1.00
    sigma_u = 0.25
    var_y = np.var(filter_y)

    t = 0 

    while t <= T - 1:
        # Draw innovations 
        inn_e = sigma_e * np.random.randn(N)
        inn_u = sigma_u * np.random.randn(N)
        s_prev = s_curr 

        for jj in range(N):
            s_curr[jj] = pphi_hat * s_prev[jj] + inn_e[jj]
            y_curr[jj] = ppsi_hat * s_curr[jj] + inn_u[jj]

        # Difference with actual data 
        v = y_curr - filter_y[t]

        for jj in range(N): 
            weights[jj] = np.exp(-(1/2) * v[jj]**2 / (0.05 * var_y))

        weights /= np.sum(weights)

        # Resample 
        state_nores[:,t] = s_curr
        filtered_state[:,t] = resample(s_curr, weights)
        s_curr = filtered_state[:, t]
        N_eff[t] = 1 / np.sum(weights**2)

        t += 1

    return filtered_state, state_nores, N_eff

filt = particle(100000, filter_y, pphi_hat, ppsi_hat)
filtered_state = filt[0]
plt.figure(1)
plt.plot(np.mean(filtered_state, axis=0), linewidth=2, label='Filtered Shocks')
plt.plot(true_s, linewidth=2, label='True Shocks')
plt.legend()
plt.title('Comparison of Filtered Shocks and True Shocks')
plt.show()

plt.figure(2)
plt.plot(ppsi_hat * np.mean(filtered_state, axis=0), linewidth=2, label='Predicted')
plt.plot(filter_y, linewidth=2, label='True Data')
plt.legend()
plt.title('Predicted y vs Actual y')
plt.show()
