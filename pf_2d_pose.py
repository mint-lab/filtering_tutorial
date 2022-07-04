import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from filterpy.monte_carlo import systematic_resample
from ekf_2d_pose import plot_results

class PFLocalizer:
    def __init__(self, v_noise_std=1, w_noise_std=1, gps_noise_std=1, dt=1,
                 N=1000, x_range=(-15, +15), y_range=(-15, +15), theta_range=(-np.pi, np.pi), v_range=(1, 10), w_range=(-np.pi/2, np.pi/2)):
        self.v_noise_std = v_noise_std
        self.w_noise_std = w_noise_std
        self.gps_noise_std = gps_noise_std
        self.dt = dt

        # Spread the initial particles uniformly
        self.pts = np.zeros((N, 5))
        self.pts[:,0] = np.random.uniform(*x_range, size=N)
        self.pts[:,1] = np.random.uniform(*y_range, size=N)
        self.pts[:,2] = np.random.uniform(*theta_range, size=N)
        self.pts[:,3] = np.random.uniform(*v_range, size=N)
        self.pts[:,4] = np.random.uniform(*w_range, size=N)
        self.weight = np.ones(N) / N

    def predict(self):
        # Move the particles
        v_noise = self.v_noise_std * np.random.randn(len(self.pts))
        w_noise = self.w_noise_std * np.random.randn(len(self.pts))
        v_delta = (self.pts[:,3] + v_noise) * self.dt
        w_delta = (self.pts[:,4] + w_noise) * self.dt
        self.pts[:,0] += v_delta * np.cos(self.pts[:,2] + w_delta / 2)
        self.pts[:,1] += v_delta * np.sin(self.pts[:,2] + w_delta / 2)
        self.pts[:,2] += w_delta
        self.pts[:,3] += v_noise
        self.pts[:,4] += w_noise

    def update(self, z):
        # Update weights of the particles
        d = np.linalg.norm(self.pts[:,0:2] - z.flatten(), axis=1)
        self.weight *= scipy.stats.norm(scale=gps_noise_std).pdf(d)
        self.weight += 1e-10
        self.weight /= sum(self.weight)

        # Resample the particles
        N = len(self.pts)
        if neff(self.weight) < N / 2:
            indices = systematic_resample(self.weight)
            self.pts[:] = self.pts[indices]
            self.weight = np.ones(N) / N

    def get_state(self):
        xy = np.average(self.pts[:,0:2], weights=self.weight, axis=0)
        c = np.average(np.cos(self.pts[:,2]), weights=self.weight)
        s = np.average(np.sin(self.pts[:,2]), weights=self.weight)
        theta = np.arctan2(s, c)
        vw = np.average(self.pts[:,3:5], weights=self.weight, axis=0)
        return np.hstack((xy, theta, vw))

def neff(weight):
    return 1. / np.sum(np.square(weight))



if __name__ == '__main__':
    # Define experimental configuration
    dt, t_end = 0.1, 8
    r, w = 10., np.pi / 4
    get_true_position = lambda t: r * np.array([[np.cos(w * t)], [np.sin(w * t)]]) # Circular motion
    get_true_heading  = lambda t: np.arctan2(np.cos(w * t), -np.sin(w * t))
    gps_noise_std = 1
    show_animation = True    # Type `%matplotlib qt5` in IPython console in Spyder to watch the animation
    show_animation_sec = 0.1 # Reduce the pause time if you want faster animation

    # Instantiate particle filter for pose (and velocity) tracking
    localizer_name = 'Particle Filter'
    localizer = PFLocalizer(v_noise_std=0.5, w_noise_std=0.5, gps_noise_std=gps_noise_std, dt=dt)

    # Prepare animation (online visualization) of the particle filter
    if show_animation:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.grid(True)
        l_pts, = ax.plot([], [], 'k.', label='Particles', alpha=0.1, markeredgewidth=0)
        l_tru, = ax.plot([], [], 'r-', label='Truth')
        l_obs, = ax.plot([], [], 'b+', label='Observation')
        l_loc, = ax.plot([], [], 'g-', label=localizer_name)
        ax.legend()

    truth, state, obser = np.empty((0, 6)), np.empty((0, 6)), np.empty((0, 3))
    for t in np.arange(0, t_end, dt):
        # Simulate position observation with additive Gaussian noise
        true_pos = get_true_position(t)
        true_ori = get_true_heading(t)
        gps_data = true_pos + np.random.normal(size=true_pos.shape, scale=gps_noise_std)

        # Predict and update the particle filter
        localizer.predict()
        localizer.update(gps_data)

        # Record true state, observation, estimated state, and its covariance
        truth = np.vstack((truth, [t] + true_pos.flatten().tolist() + [true_ori, r * w, w]))
        state = np.vstack((state, [t] + localizer.get_state().flatten().tolist()))
        obser = np.vstack((obser, [t] + gps_data.flatten().tolist()))

        # Draw animation of the particle filter
        if show_animation:
            l_pts.set_data(localizer.pts[:,0], localizer.pts[:,1])
            l_tru.set_data(truth[:,1], truth[:,2])
            l_obs.set_data(obser[:,1], obser[:,2])
            l_loc.set_data(state[:,1], state[:,2])
            plt.draw()
            plt.pause(show_animation_sec)

    # Visualize the results
    plt.ioff()
    plot_results(localizer_name, truth, state, obser)
    plt.show()