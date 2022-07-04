import numpy as np
from ekf_2d_pose import EKFLocalizer, plot_results

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
    localizer_name = 'EKF+OffCentered'
    localizer = EKFLocalizerOC(v_noise_std=0.1, w_noise_std=0.1, gps_noise_std=gps_noise_std, gps_offset=gps_offset, dt=dt)

    truth, state, obser, covar = [], [], [], []
    for t in np.arange(0, t_end, dt):
        # Simulate position observation with off-centered GPS and additive Gaussian noise
        true_pos = get_true_position(t)
        true_ori = get_true_heading(t)
        R = np.array([[np.cos(true_ori), -np.sin(true_ori)], [np.sin(true_ori), np.cos(true_ori)]])
        gps_data = true_pos + R @ gps_offset + np.random.normal(size=true_pos.shape, scale=gps_noise_std)

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