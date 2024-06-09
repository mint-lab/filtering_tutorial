import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.stats import plot_covariance

class EKFLocalizer(ExtendedKalmanFilter):
    def __init__(self, v_noise_std=1, w_noise_std=1, gps_noise_std=1, dt=1):
        super().__init__(dim_x=5, dim_z=2)
        self.motion_noise = np.array([[v_noise_std * v_noise_std, 0], [0, w_noise_std * w_noise_std]])
        self.h = lambda x: x[0:2]
        self.H = lambda x: np.eye(2, 5)
        self.R = gps_noise_std * gps_noise_std * np.eye(2)
        self.dt = dt

    def predict(self):
        x, y, theta, v, w = self.x.flatten()
        vt, wt = v * self.dt, w * self.dt
        s, c = np.sin(theta + wt / 2), np.cos(theta + wt / 2)

        # Predict the state
        self.x[0] = x + vt * c
        self.x[1] = y + vt * s
        self.x[2] = theta + wt
        #self.x[3] = v # Not necessary
        #self.x[4] = w # Not necessary

        # Predict the covariance
        self.F = np.array([
            [1, 0, -vt * s, self.dt * c, -vt * self.dt * s / 2],
            [0, 1,  vt * c, self.dt * s,  vt * self.dt * c / 2],
            [0, 0,       1,           0,               self.dt],
            [0, 0,       0,           1,                     0],
            [0, 0,       0,           0,                     1]])
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

def plot_results(localizer_name, truth, state, obser=None, covar=None, covar_step=5, covar_sigma=0.98): # 0.98 means 2 x sigma.
    truth = np.array(truth)
    state = np.array(state)
    if obser is not None:
        obser = np.array(obser)
    if covar is not None:
        covar = np.array(covar)
    state_dim = len(state[0]) - 1

    plt.figure('XY')
    plt.plot(truth[:,1], truth[:,2], 'r-', label='Truth')
    if obser is not None:
        plt.plot(obser[:,1], obser[:,2], 'b+', label='Observation')
    plt.plot(state[:,1], state[:,2], 'g-', label=localizer_name)
    if covar is not None:
        for i, cov in enumerate(covar):
            if i % covar_step == 0:
                plot_covariance(state[i][1:3], cov[1:].reshape(state_dim, state_dim)[0:2,0:2], interval=covar_sigma, edgecolor='g', alpha=0.5)
    plt.axis('equal')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.grid()
    plt.legend()

    plt.figure('Time-X')
    plt.plot(truth[:,0], truth[:,1], 'r-', label='Truth')
    if obser is not None:
        plt.plot(obser[:,0], obser[:,1], 'b+', label='Observation')
    plt.plot(state[:,0], state[:,1], 'g-', label=localizer_name)
    plt.xlabel('Time [sec]')
    plt.ylabel('X [m]')
    plt.grid()
    plt.legend()

    plt.figure('Time-Y')
    plt.plot(truth[:,0], truth[:,2], 'r-', label='Truth')
    if obser is not None:
        plt.plot(obser[:,0], obser[:,2], 'b+', label='Observation')
    plt.plot(state[:,0], state[:,2], 'g-', label=localizer_name)
    plt.xlabel('Time [sec]')
    plt.ylabel('Y [m]')
    plt.grid()
    plt.legend()

    if state_dim >= 3:
        plt.figure('Time-Theta')
        plt.plot(truth[:,0], np.rad2deg(truth[:,3]), 'r-', label='Truth')
        plt.plot(state[:,0], np.rad2deg(state[:,3]), 'g-', label=localizer_name)
        plt.xlabel('Time [sec]')
        plt.ylabel(r'Orientaiton $\theta$ [deg]')
        plt.grid()
        plt.legend()

    if state_dim >= 4:
        plt.figure('Time-V')
        plt.plot(truth[:,0], truth[:,4], 'r-', label='Truth')
        plt.plot(state[:,0], state[:,4], 'g-', label=localizer_name)
        plt.xlabel('Time [sec]')
        plt.ylabel('Linear Velocity $v$ [m/s]')
        plt.grid()
        plt.legend()

    if state_dim >= 5:
        plt.figure('Time-W')
        plt.plot(truth[:,0], np.rad2deg(truth[:,5]), 'r-', label='Truth')
        plt.plot(state[:,0], np.rad2deg(state[:,5]), 'g-', label=localizer_name)
        plt.xlabel('Time [sec]')
        plt.ylabel('Angular Velocity $w$ [deg/s]')
        plt.grid()
        plt.legend()



if __name__ == '__main__':
    # Define experimental configuration
    dt, t_end = 0.1, 8
    r, w = 10., np.pi / 4
    get_true_position = lambda t: r * np.array([[np.cos(w * t)], [np.sin(w * t)]]) # Circular motion
    get_true_heading  = lambda t: np.arctan2(np.cos(w * t), -np.sin(w * t))
    gps_noise_std = 1

    # Instantiate EKF for pose (and velocity) tracking
    localizer_name = 'EKF'
    localizer = EKFLocalizer(v_noise_std=0.1, w_noise_std=0.1, gps_noise_std=gps_noise_std, dt=dt)

    truth, state, obser, covar = [], [], [], []
    for t in np.arange(0, t_end, dt):
        # Simulate position observation with additive Gaussian noise
        true_pos = get_true_position(t)
        true_ori = get_true_heading(t)
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