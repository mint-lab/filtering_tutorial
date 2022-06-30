import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.stats import plot_covariance

def fx(state, dt):
    x, y, theta, v, w = state.flatten()
    vt, wt = v * dt, w * dt
    s, c = np.sin(theta + wt / 2), np.cos(theta + wt / 2)

    # Predict the state
    fx = np.array([
        [x + vt * c],
        [y + vt * s],
        [theta + wt],
        [v],
        [w]])
    if fx[2,0] >= np.pi:
        fx[2,0] -= 2 * np.pi
    elif fx[2,0] < -np.pi:
        fx[2,0] += 2 * np.pi
    return fx

def Fx(state, dt):
    x, y, theta, v, w = state.flatten()
    vt, wt = v * dt, w * dt
    s, c = np.sin(theta + wt / 2), np.cos(theta + wt / 2)

    # Predict the covariance
    Fx = np.array([
        [1, 0, -vt * s, dt * c, -vt * dt * s / 2],
        [0, 1,  vt * c, dt * s,  vt * dt * c / 2],
        [0, 0,       1,      0,               dt],
        [0, 0,       0,      1,                0],
        [0, 0,       0,      0,                1]])
    return Fx

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
    ekf = ExtendedKalmanFilter(dim_x=5, dim_z=2)
    ekf.Q = 0.1 * np.eye(5)
    ekf.R = gps_noise_std * gps_noise_std * np.eye(2)

    record = []
    for t in np.arange(0, t_end, dt):
        # Simulate position observation with additive Gaussian noise
        true_pos = get_true_position(t)
        true_ori = get_true_heading(t)
        obs = true_pos + np.random.normal(size=true_pos.shape, scale=gps_noise_std)

        # Predict and update the EKF
        ekf.F = Fx(ekf.x, dt)
        ekf.x = fx(ekf.x, dt)
        ekf.predict()
        ekf.update(obs, Hx, hx)

        record.append([t] + true_pos.flatten().tolist() + [true_ori] + obs.flatten().tolist() + ekf.x.flatten().tolist() + ekf.P.flatten().tolist())
    record = np.array(record)

    # Visualize the results
    plt.figure()
    plt.plot(record[:,1], record[:,2], 'r-', label='Truth')
    plt.plot(record[:,4], record[:,5], 'b+', label='Observation')
    plt.plot(record[:,6], record[:,7], 'g-', label='EKF')
    for i, line in enumerate(record):
        if i % 5 == 0:
            plot_covariance(line[6:8], line[11:].reshape(5, 5)[0:2,0:2], interval=0.98, edgecolor='g', alpha=0.5)
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(record[:,0], record[:,1], 'r-', label='Truth')
    plt.plot(record[:,0], record[:,4], 'b+', label='Observation')
    plt.plot(record[:,0], record[:,6], 'g-', label='EKF')
    plt.xlabel('Time')
    plt.ylabel('X')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(record[:,0], record[:,2], 'r-', label='Truth')
    plt.plot(record[:,0], record[:,5], 'b+', label='Observation')
    plt.plot(record[:,0], record[:,7], 'g-', label='EKF')
    plt.xlabel('Time')
    plt.ylabel('Y')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(record[:,0], record[:,3] * 180 / np.pi, 'r-', label='Truth')
    plt.plot(record[:,0], record[:,8] * 180 / np.pi, 'g-', label='EKF')
    plt.xlabel('Time')
    plt.ylabel(r'Orientaiton $\theta$ [deg]')
    plt.grid()
    plt.legend()