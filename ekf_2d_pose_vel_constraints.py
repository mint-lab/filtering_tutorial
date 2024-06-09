import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
from ekf_2d_pose import EKFLocalizer, plot_results

class EKFLocalizerVC(ExtendedKalmanFilter):
    def __init__(self, v_noise_std=1, w_noise_std=1, gps_noise_std=1, dt=1, v_epsilon=0.001, w_max=1):
        super().__init__(dim_x=5, dim_z=2)
        self.motion_noise = np.array([[v_noise_std * v_noise_std, 0], [0, w_noise_std * w_noise_std]])
        self.h = lambda x: x[0:2]
        self.H = lambda x: np.eye(2, 5)
        self.R = gps_noise_std * gps_noise_std * np.eye(2)
        self.dt = dt
        self.v_epsilon = v_epsilon
        self.w_max = w_max

    def predict(self):
        x, y, theta, v, w = self.x.flatten()
        vt, wt = v * self.dt, w * self.dt
        s, c = np.sin(theta + wt / 2), np.cos(theta + wt / 2)
        w_sat = self.w_max * np.tanh(w / self.w_max)

        # Predict the state
        self.x[0] = x + vt * c
        self.x[1] = y + vt * s
        self.x[2] = theta + wt
        #self.x[3] = v # Not necessary
        self.x[4] = w_sat # Constraint #2) Angular rate saturation

        # Predict the covariance
        self.F = np.array([
            [1, 0, -vt * s, self.dt * c, -vt * self.dt * s / 2],
            [0, 1,  vt * c, self.dt * s,  vt * self.dt * c / 2],
            [0, 0,       1,           0,               self.dt],
            [0, 0,       0,           1,                     0],
            [0, 0,       0,           0,     1 - w_sat * w_sat]]) # Constraint #2) Angular rate saturation
        W = np.array([
            [self.dt * c, -vt * self.dt * s / 2],
            [self.dt * s,  vt * self.dt * c / 2],
            [0, self.dt],
            [1, 0],
            [0, 1]])
        self.Q = W @ self.motion_noise @ W.T
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, z):
        super().update(z, HJacobian=self.H, Hx=self.h, R=self.R)
        if self.v_epsilon > 0 and self.x[3] < self.v_epsilon:
            # Constraint #1) Heading angle correction
            self.x[2] += np.pi
            self.x[3] *= -1



if __name__ == '__main__':
    # Define experimental configuration
    dt, t_end = 0.1, 8
    r, w = 10., np.pi / 4
    get_true_position = lambda t: r * np.array([[np.cos(w * t)], [np.sin(w * t)]]) # Circular motion
    get_true_heading  = lambda t: np.arctan2(np.cos(w * t), -np.sin(w * t))
    gps_noise_std = 1
    gps_outlier_deadzone = (-12, -8, -5, 5) # (x_min, x_max, y_min, y_max)
    gps_outlier_noise_std = 10
    is_inside_deadzone = lambda x, y, x_min, x_max, y_min, y_max: (x_min <= x <= x_max) and (y_min <= y <= y_max)

    # Note) Please compare the results by `EKFLocalizer` and `EKFLocalizerVC`
    np.random.seed(0) # For reproducibility

    # Instantiate EKF for pose (and velocity) tracking
    localizer_name = 'EKF+VelocityConstraints'
    localizer = EKFLocalizerVC(v_noise_std=0.1, w_noise_std=0.1, gps_noise_std=gps_noise_std, dt=dt)

    truth, state, obser, covar = [], [], [], []
    for t in np.arange(0, t_end, dt):
        # Simulate position observation with additive Gaussian noise
        true_pos = get_true_position(t)
        true_ori = get_true_heading(t)
        if is_inside_deadzone(*true_pos, *gps_outlier_deadzone):
            gps_data = true_pos + np.random.normal(size=true_pos.shape, scale=gps_outlier_noise_std)
        else:
            gps_data = true_pos + np.random.normal(size=true_pos.shape, scale=gps_noise_std)

        # Predict and update the EKF
        localizer.predict()
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