import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

if __name__ == '__main__':
    # Prepare a noisy signal
    true_signal = lambda t: 10 * np.sin(2*np.pi/2*t) * np.cos(2*np.pi/10*t)
    times = np.arange(1, 10, 0.01)
    truth = true_signal(times)
    obs_signal = truth + np.random.normal(scale=1, size=truth.size)

    # Instantiate Kalman filter for noise filtering
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.F = np.eye(1)
    kf.H = np.eye(1)
    kf.P = 10 * np.eye(1)
    kf.Q = 0.02 * np.eye(1)
    kf.R = 1 * np.eye(1)

    xs = []
    for z in obs_signal:
        # Predict and update the Kalman filter
        kf.predict()
        kf.update(z)
        xs.append(kf.x.flatten())

    # Visualize the results
    plt.figure()
    plt.plot(times, obs_signal, 'b-',  label='Observation', alpha=0.3)
    plt.plot(times, xs,         'g-',  label='Kalman Filter')
    plt.plot(times, truth,      'r-',  label='Truth', alpha=0.5)
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    plt.show()