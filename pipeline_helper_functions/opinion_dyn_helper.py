import pandas as pd
import plotly.express as px
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
import plotly.io as pio
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

# Computes the density of GMM
def gmm_density(x_input, y_input, gmm):

    x_flat = np.ravel(x_input)
    y_flat = np.ravel(y_input)
    points = np.column_stack([x_flat, y_flat])
    
    density_vals = np.zeros(len(points))
    
    # Sum the weighted PDF from each GMM component
    for weight, mean, cov in zip(gmm.weights_, gmm.means_, gmm.covariances_):
        rv = multivariate_normal(mean=mean, cov=cov)
        density_vals += weight * rv.pdf(points)
    
    return density_vals.reshape(np.shape(x_input))

# Computes log-density of a GMM (potential)
def gmm_density_and_loggrad(x_input, y_input, gmm):
    x_flat = np.ravel(x_input)
    y_flat = np.ravel(y_input)
    points = np.column_stack([x_flat, y_flat])
    N = len(points)

    density_vals = np.zeros(N)
    grad = np.zeros_like(points)

    # Iterate through each GMM component
    for weight, mean, cov in zip(gmm.weights_, gmm.means_, gmm.covariances_):
        rv = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
        pdf_vals = rv.pdf(points)
        diff = points - mean
        inv_cov = np.linalg.pinv(cov) 
        grad_comp = -pdf_vals[:, None] * (diff @ inv_cov.T)

        density_vals += weight * pdf_vals
        grad += weight * grad_comp

    eps = 1e-9
    grad_log_density = grad / (density_vals[:, None] + eps)

    return grad_log_density

# Boundary conditions
def reflect(val, low, high):
    range_size = high - low
    val_shifted = (val - low) % (2 * range_size)
    reflected = np.where(val_shifted < range_size, val_shifted, 2 * range_size - val_shifted)
    return reflected + low

# Actual model
def run_simulation(data, T, sigma_noise, gmm_components, alpha, beta, gamma, random_seed=42):
    
    np.random.seed(random_seed)
    
    D, N = data.shape
    history = [data.copy()]

    for t in range(T):
        X_t = history[-1]

        # Small noise to avoid GMM fitting issues when multiple points are duplicate
        X_t_noisy = X_t.T + np.random.normal(scale=1e-6, size=(N, D))

        # Fit GMM
        gmm = GaussianMixture(n_components=gmm_components, covariance_type='full', reg_covar=1e-2)
        gmm.fit(X_t_noisy)

        # Distances for weight
        distances = cdist(X_t_noisy, X_t_noisy, metric='euclidean')
        W = np.exp(-distances ** 2)
        W /= W.sum(axis=1, keepdims=True)

        weighted_sum = W @ X_t_noisy 
        
        # Gradient effect
        F_x = gmm_density_and_loggrad(X_t[0, :], X_t[1, :], gmm) 

        # White noise effect
        noise = np.random.normal(0, sigma_noise, size=(N, D))

        X_next = alpha * weighted_sum - beta * F_x + gamma * noise

        X_next = np.clip(X_next, -3, 3)

        for dim in range(D):
            mask_low = X_next[:, dim] <= -3
            X_next[mask_low, dim] = -3 + (-3 - X_next[mask_low, dim])
            mask_high = X_next[:, dim] >= 3
            X_next[mask_high, dim] = 3 - (X_next[mask_high, dim] - 3)

        history.append(X_next.T)

    final_positions = history[-1]
    return final_positions


# Just some plotting
def plot_with_simulation_separate(concatenated_df, simulation_points):

    print("Data ranges and checks:")
    print("Opposition to Immigration Scaled min/max:", concatenated_df['Opposition to Immigration Scaled'].min(), concatenated_df['Opposition to Immigration Scaled'].max())
    print("Welfare State Scaled min/max:", concatenated_df['Welfare State Scaled'].min(), concatenated_df['Welfare State Scaled'].max())
    
    sim_x = np.array(simulation_points[0])
    sim_y = np.array(simulation_points[1])
    
    print("Simulation X min/max:", np.min(sim_x), np.max(sim_x))
    print("Simulation Y min/max:", np.min(sim_y), np.max(sim_y))
    
    print("Any NaNs or infs in simulation X?", np.isnan(sim_x).any(), np.isinf(sim_x).any())
    print("Any NaNs or infs in simulation Y?", np.isnan(sim_y).any(), np.isinf(sim_y).any())
    
    def clip_data(arr, min_val=-1e3, max_val=1e3):
        arr = np.clip(arr, min_val, max_val)
        return arr
    
    sim_x = clip_data(sim_x)
    sim_y = clip_data(sim_y)
    
    fig = px.scatter(
        concatenated_df,
        x='Opposition to Immigration Scaled',
        y='Welfare State Scaled',
        color='Label',
        symbol='Label'
    )
    
    fig.add_scatter(
        x=sim_x,
        y=sim_y,
        mode='markers',
        marker=dict(
            color='rgba(0,0,0,0.2)',
            size=4,
            symbol='circle'
        ),
        name='Simulation Points'
    )
    
    xmin = min(concatenated_df['Opposition to Immigration Scaled'].min(), np.min(sim_x))
    xmax = max(concatenated_df['Opposition to Immigration Scaled'].max(), np.max(sim_x))
    ymin = min(concatenated_df['Welfare State Scaled'].min(), np.min(sim_y))
    ymax = max(concatenated_df['Welfare State Scaled'].max(), np.max(sim_y))
    
    padding_x = (xmax - xmin) * 0.1
    padding_y = (ymax - ymin) * 0.1
    
    fig.update_layout(
        title='Scaled Positions with Simulation Overlay',
        xaxis=dict(range=[xmin - padding_x, xmax + padding_x]),
        yaxis=dict(range=[ymin - padding_y, ymax + padding_y]),
    )
    
    return fig
