import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from ekf_2d_pose import plot_results

def fx(state, dt):
    x, y, theta, v, w = state.flatten()
    vt, wt = v * dt, w * dt
    s, c = np.sin(theta + wt / 2), np.cos(theta + wt / 2)
    return np.array([
        [x + vt * c],
        [y + vt * s],
        [theta + wt],
        [v],
        [w]])

def Fx(state, dt):
    x, y, theta, v, w = state.flatten()
    vt, wt = v * dt, w * dt
    s, c = np.sin(theta + wt / 2), np.cos(theta + wt / 2)
    return np.array([
        [1, 0, -vt * s, dt * c, -vt * dt * s / 2],
        [0, 1,  vt * c, dt * s,  vt * dt * c / 2],
        [0, 0,       1,      0,               dt],
        [0, 0,       0,      1,                0],
        [0, 0,       0,      0,                1]])

def hx(state):
    x, y, _, _, _ = state.flatten()
    return np.array([[x], [y]])

def Hx(state):
    return np.eye(2, 5)



if __name__ == '__main__':
    # Define experimental configuration
    dt, t_end = 0.1, 8
    r, w = 10., np.pi / 4
    get_true_position = lambda t: r * np.array([[np.cos(w * t)], [np.sin(w * t)]]) # Circular motion
    get_true_heading  = lambda t: np.arctan2(np.cos(w * t), -np.sin(w * t))
    gps_noise_std = 1

    # Instantiate EKF for pose (and velocity) tracking
    localizer_name = 'EKF+SimpleNoise'
    localizer = ExtendedKalmanFilter(dim_x=5, dim_z=2)
    localizer.Q = 0.1 * np.eye(5)
    localizer.R = gps_noise_std * gps_noise_std * np.eye(2)

    truth, state, obser, covar = [], [], [], []
    for t in np.arange(0, t_end, dt):
        # Simulate position observation with additive Gaussian noise
        true_pos = get_true_position(t)
        true_ori = get_true_heading(t)
        gps_data = true_pos + np.random.normal(size=true_pos.shape, scale=gps_noise_std)

        # Predict and update the EKF
        localizer.F = Fx(localizer.x, dt)
        localizer.x = fx(localizer.x, dt)
        localizer.predict()
        localizer.update(gps_data, Hx, hx)

        if localizer.x[2] >= np.pi:
            localizer.x[2] -= 2 * np.pi
        elif localizer.x[2] < -np.pi:
            localizer.x[2] += 2 * np.pi

        # Record true state, observation, estimated state, and its covariance
        truth.append([t] + true_pos.flatten().tolist() + [true_ori, r * w, w])
        state.append([t] + localizer.x.flatten().tolist())
        obser.append([t] + gps_data.flatten().tolist())
        covar.append([t] + localizer.P.flatten().tolist())

    # Visualize the results
    plot_results(localizer_name, truth, state, obser, covar)