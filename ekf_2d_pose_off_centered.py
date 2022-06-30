import numpy as np
import matplotlib.pyplot as plt
from ekf_2d_pose import EKFLocalizer
from filterpy.stats import plot_covariance

class EKFLocalizerOC(EKFLocalizer):
    def __init__(self, v_noise_std=1, w_noise_std=1, gps_noise_std=1, gps_offset=(0,0), dt=1):
        super().__init__(v_noise_std, w_noise_std, gps_noise_std, dt)
        self.h = self.hx
        self.H = self.Hx
        self.gps_offset_x, self.gps_offset_y = gps_offset.flatten()

    def hx(self, state):
        x, y, theta, v, w = state.flatten()
        s, c = np.sin(theta), np.cos(theta)
        return np.array([
            [x + self.gps_offset_x * c - self.gps_offset_y * s],
            [y + self.gps_offset_x * s + self.gps_offset_y * c]])

    def Hx(self, state):
        x, y, theta, v, w = state.flatten()
        s, c = np.sin(theta), np.cos(theta)
        return np.array([
            [1, 0, -self.gps_offset_x * s - self.gps_offset_y * c, 0, 0],
            [0, 1,  self.gps_offset_x * c - self.gps_offset_y * s, 0, 0]])



if __name__ == '__main__':
    # Define experimental configuration
    dt, t_end = 0.1, 8
    r, w = 10., np.pi / 4
    get_true_position = lambda t: r * np.array([[np.cos(w * t)], [np.sin(w * t)]]) # Circular motion
    get_true_heading  = lambda t: np.arctan2(np.cos(w * t), -np.sin(w * t))
    gps_noise_std = 1
    gps_offset = np.array([[1], [0]])

    # Instantiate EKF for pose (and velocity) tracking
    ekf = EKFLocalizerOC(v_noise_std=1, w_noise_std=0.1, gps_noise_std=gps_noise_std, gps_offset=gps_offset, dt=dt)

    record = []
    for t in np.arange(0, t_end, dt):
        # Simulate position observation with off-centered GPS and additive Gaussian noise
        true_pos = get_true_position(t)
        true_ori = get_true_heading(t)
        R = np.array([[np.cos(true_ori), -np.sin(true_ori)], [np.sin(true_ori), np.cos(true_ori)]])
        obs = true_pos + R @ gps_offset + np.random.normal(size=true_pos.shape, scale=gps_noise_std)

        # Predict and update the EKF
        ekf.predict()
        ekf.update(obs)

        if ekf.x[2] >= np.pi:
            ekf.x[2] -= 2 * np.pi
        elif ekf.x[2] < -np.pi:
            ekf.x[2] += 2 * np.pi

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