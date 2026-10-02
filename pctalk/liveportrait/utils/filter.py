
import torch
import numpy as np
from pykalman import KalmanFilter


def smooth(x_d_lst, shape, device, observation_variance=3e-7, process_variance=1e-5):
    print("smooth using device:", device)
    if device.startswith('cuda'):
        return smooth_gpu(x_d_lst, shape, device, observation_variance, process_variance)
    x_d_lst_reshape = [x.reshape(-1) for x in x_d_lst]
    x_d_stacked = np.vstack(x_d_lst_reshape)
    kf = KalmanFilter(
        initial_state_mean=x_d_stacked[0],
        n_dim_obs=x_d_stacked.shape[1],
        transition_covariance=process_variance * np.eye(x_d_stacked.shape[1]),
        observation_covariance=observation_variance * np.eye(x_d_stacked.shape[1])
    )
    smoothed_state_means, _ = kf.smooth(x_d_stacked)
    x_d_lst_smooth = [torch.tensor(state_mean.reshape(shape[-2:]), dtype=torch.float32, device=device) for state_mean in smoothed_state_means]
    return x_d_lst_smooth


def smooth_gpu(x_d_lst, shape, device, observation_variance=3e-7, process_variance=1e-5):
    """
    Kalman Smoothing using PyTorch on GPU
    Args:
        x_d_lst: list[torch.Tensor]  # 每一帧数据
        shape: tuple                 # 原始形状 (batch, H, W)
        device: torch.device         # 'cuda' or 'cpu'
        observation_variance: float
        process_variance: float
    Returns:
        x_d_lst_smooth: list[torch.Tensor]  # 平滑后的数据
    """

    # Flatten the sequence to [T, D].
    if device.startswith('cuda'):
        device=torch.device(device)
    x_d_tensors = [
        torch.as_tensor(x, dtype=torch.float32, device=device).reshape(-1)
        for x in x_d_lst
    ]
    x_d_stacked = torch.stack(x_d_tensors)  # [T, D]
    T, D = x_d_stacked.shape

    # Reuse constant covariance matrices across frames.
    I = torch.eye(D, device=device)
    transition_cov = process_variance * I
    observation_cov = observation_variance * I

    # Initialize the state from the first observation.
    state_mean = x_d_stacked[0]
    state_cov = observation_cov.clone()

    filtered_means = []
    filtered_covs = []

    # Kalman filter forward pass.
    for t in range(T):
        obs = x_d_stacked[t]

        # Solve for the Kalman gain without forming an explicit inverse.
        S = state_cov + observation_cov  # [D, D]
        K = torch.linalg.solve(S, state_cov.T).T  # 等价于 state_cov @ S^-1

        # Update the filtered state.
        state_mean = state_mean + K @ (obs - state_mean)
        state_cov = (I - K) @ state_cov

        filtered_means.append(state_mean)
        filtered_covs.append(state_cov)

        # Predict the next state.
        state_cov = state_cov + transition_cov

    # Rauch-Tung-Striebel smoothing pass.
    smoothed_means = [None] * T
    smoothed_means[-1] = filtered_means[-1]

    for t in reversed(range(T - 1)):
        cov_t = filtered_covs[t]
        # Compute the smoother gain.
        G = torch.linalg.solve(cov_t + transition_cov, cov_t.T).T
        smoothed_means[t] = filtered_means[t] + G @ (smoothed_means[t + 1] - filtered_means[t])

    # Restore the original sequence shape.
    x_d_lst_smooth = [
        smoothed_means[i].reshape(shape[-2:]).to(device)
        for i in range(T)
    ]

    return x_d_lst_smooth
