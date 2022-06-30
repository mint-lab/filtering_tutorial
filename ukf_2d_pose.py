import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.stats import plot_covariance

class UKFLocalizer(UnscentedKalmanFilter):
    def __init__(self, v_noise_std=1, w_noise_std=1, gps_noise_std=1, dt=1):
        self.sigma_points = MerweScaledSigmaPoints(5, alpha=.1, beta=2., kappa=-1)
        super().__init__(dim_x=5, dim_z=2, dt=dt, fx=self.fx, hx=self.hx, points=self.sigma_points)
        vv = v_noise_std * v_noise_std
        vw = v_noise_std * w_noise_std
        ww = w_noise_std * w_noise_std
        self.motion_noise = np.array([[vv, vw], [vw, ww]])
        self.R = gps_noise_std * gps_noise_std * np.eye(2)
        self.dt = dt

    def fx(self, state, dt):
        x, y, theta, v, w = state.flatten()
        vt, wt = v * dt, w * dt
        s, c = np.sin(theta + wt / 2), np.cos(theta + wt / 2)
        return np.array([
            x + vt * c,
            y + vt * s,
            theta + wt,
            v,
            w])

    def hx(self, state):
        x, y, _, _, _ = state.flatten()
        return np.array([x, y])

    def predict(self):
        x, y, theta, v, w = self.x.flatten()
        vt, wt = v * self.dt, w * self.dt
        s, c = np.sin(theta + wt / 2), np.cos(theta + wt / 2)

        # Set the covariance of transition noise
        W = np.array([
            [self.dt * c, -vt * self.dt * s / 2],
            [self.dt * s,  vt * self.dt * c / 2],
            [0, self.dt],
            [1, 0],
            [0, 1]])
        self.Q = W @ self.motion_noise @ W.T

        super().predict()

    def update(self, z):
        super().update(z.flatten())


if __name__ == '__main__':
    # Define experimental configuration
    dt, t_end = 0.1, 8
    r, w = 10., np.pi / 4
    get_true_position = lambda t: r * np.array([[np.cos(w * t)], [np.sin(w * t)]]) # Circular motion
    get_true_heading  = lambda t: np.arctan2(np.cos(w * t), -np.sin(w * t))
    gps_noise_std = 1

    # Instantiate UKF for pose (and velocity) tracking
    ukf = UKFLocalizer(v_noise_std=1, w_noise_std=0.1, gps_noise_std=gps_noise_std, dt=dt)

    record = []
    for t in np.arange(0, t_end, dt):
        # Simulate position observation with additive Gaussian noise
        true_pos = get_true_position(t)
        true_ori = get_true_heading(t)
        obs = true_pos + np.random.normal(size=true_pos.shape, scale=gps_noise_std)

        # Predict and update the UKF
        ukf.predict()
        ukf.update(obs)

        if ukf.x[2] >= np.pi:
            ukf.x[2] -= 2 * np.pi
        elif ukf.x[2] < -np.pi:
            ukf.x[2] += 2 * np.pi

        record.append([t] + true_pos.flatten().tolist() + [true_ori] + obs.flatten().tolist() + ukf.x.flatten().tolist() + ukf.P.flatten().tolist())
    record = np.array(record)

    # Visualize the results
    plt.figure()
    plt.plot(record[:,1], record[:,2], 'r-', label='Truth')
    plt.plot(record[:,4], record[:,5], 'b+', label='Observation')
    plt.plot(record[:,6], record[:,7], 'g-', label='UKF')
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
    plt.plot(record[:,0], record[:,6], 'g-', label='UKF')
    plt.xlabel('Time')
    plt.ylabel('X')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(record[:,0], record[:,2], 'r-', label='Truth')
    plt.plot(record[:,0], record[:,5], 'b+', label='Observation')
    plt.plot(record[:,0], record[:,7], 'g-', label='UKF')
    plt.xlabel('Time')
    plt.ylabel('Y')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(record[:,0], record[:,3] * 180 / np.pi, 'r-', label='Truth')
    plt.plot(record[:,0], record[:,8] * 180 / np.pi, 'g-', label='UKF')
    plt.xlabel('Time')
    plt.ylabel(r'Orientaiton $\theta$ [deg]')
    plt.grid()
    plt.legend()