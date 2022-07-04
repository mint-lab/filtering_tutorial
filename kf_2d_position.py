import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

if __name__ == '__main__':
    # Define experimental configuration
    dt, t_end = 0.1, 8
    r, w = 10., np.pi / 4
    get_true_position = lambda t: r * np.array([[np.cos(w * t)], [np.sin(w * t)]]) # Circular motion
    gps_noise_std = 1

    # Instantiate Kalman filter for position tracking
    localizer_name = 'Kalman Filter'
    localizer = KalmanFilter(dim_x=2, dim_z=2)
    localizer.F = np.eye(2)
    localizer.H = np.eye(2)
    localizer.Q = 0.1 * np.eye(2)
    localizer.R = gps_noise_std * gps_noise_std * np.eye(2)

    record = []
    for t in np.arange(0, t_end, dt):
        # Simulate position observation with additive Gaussian noise
        truth = get_true_position(t)
        obs = truth + np.random.normal(size=truth.shape, scale=gps_noise_std)

        # Predict and update the Kalman filter
        localizer.predict()
        localizer.update(obs)

        record.append([t] + truth.flatten().tolist() + obs.flatten().tolist() + localizer.x.flatten().tolist() + localizer.P.flatten().tolist())
    record = np.array(record)

    # Visualize the results
    plt.figure()
    plt.plot(record[:,1], record[:,2], 'r-', label='Truth')
    plt.plot(record[:,3], record[:,4], 'b+', label='Observation')
    plt.plot(record[:,5], record[:,6], 'g-', label=localizer_name)
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(record[:,0], record[:,1], 'r-', label='Truth')
    plt.plot(record[:,0], record[:,3], 'b+', label='Observation')
    plt.plot(record[:,0], record[:,5], 'g-', label=localizer_name)
    plt.xlabel('Time')
    plt.ylabel('X')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(record[:,0], record[:,2], 'r-', label='Truth')
    plt.plot(record[:,0], record[:,4], 'b+', label='Observation')
    plt.plot(record[:,0], record[:,6], 'g-', label=localizer_name)
    plt.xlabel('Time')
    plt.ylabel('Y')
    plt.grid()
    plt.legend()

    plt.show()