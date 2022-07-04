import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from ekf_2d_pose import plot_results

class EKFLocalizerOD(ExtendedKalmanFilter):
    def __init__(self, v_noise_std=1, w_noise_std=1, gps_noise_std=1, dt=1):
        super().__init__(dim_x=3, dim_z=2)
        self.motion_noise = np.array([[v_noise_std * v_noise_std, 0], [0, w_noise_std * w_noise_std]])
        self.h = lambda x: x[0:2]
        self.H = lambda x: np.eye(2, 3)
        self.R = gps_noise_std * gps_noise_std * np.eye(2)
        self.dt = dt

    def predict(self, u):
        x, y, theta = self.x.flatten()
        v, w = u.flatten()
        vt, wt = v * self.dt, w * self.dt
        s, c = np.sin(theta + wt / 2), np.cos(theta + wt / 2)

        # Predict the state
        self.x[0] = x + vt * c
        self.x[1] = y + vt * s
        self.x[2] = theta + wt

        # Predict the covariance
        self.F = np.array([
            [1, 0, -vt * s],
            [0, 1,  vt * c],
            [0, 0,       1]])
        W = np.array([
            [self.dt * c, -vt * self.dt * s / 2],
            [self.dt * s,  vt * self.dt * c / 2],
            [0, self.dt]])
        self.Q = W @ self.motion_noise @ W.T
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, z):
        super().update(z, HJacobian=self.H, Hx=self.h, R=self.R)



if __name__ == '__main__':
    # Define experimental configuration
    dt, t_end = 0.1, 8
    r, w = 10., np.pi / 4
    get_true_position = lambda t: r * np.array([[np.cos(w * t)], [np.sin(w * t)]]) # Circular motion
    get_true_heading  = lambda t: np.arctan2(np.cos(w * t), -np.sin(w * t))
    gps_noise_std = 1

    # Instantiate EKF for pose tracking
    localizer_name = 'EKF+Odometry'
    localizer = EKFLocalizerOD(v_noise_std=0.2, w_noise_std=0.2, gps_noise_std=gps_noise_std, dt=dt)

    truth, state, obser, covar = [], [], [], []
    for t in np.arange(0, t_end, dt):
        # Simulate position observation (with additive Gaussian noise) and velocity observation (with slippage loss)
        true_pos = get_true_position(t)
        true_ori = get_true_heading(t)
        gps_data = true_pos + np.random.normal(size=true_pos.shape, scale=gps_noise_std)
        odo_data = np.array([r * w * 0.95, w * 0.90])

        # Predict and update the EKF
        localizer.predict(odo_data)
        localizer.update(gps_data)

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
    plt.show()