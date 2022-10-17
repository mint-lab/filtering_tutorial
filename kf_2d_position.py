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

    times, truth, zs, xs, = [], [], [], []
    for t in np.arange(0, t_end, dt):
        # Simulate position observation with additive Gaussian noise
        true = get_true_position(t)
        z = true + np.random.normal(size=true.shape, scale=gps_noise_std)

        # Predict and update the Kalman filter
        localizer.predict()
        localizer.update(z)

        times.append(t)
        truth.append(true.flatten())
        zs.append(z.flatten())
        xs.append(localizer.x.flatten())
    times, truth, zs, xs = np.array(times), np.array(truth), np.array(zs), np.array(xs)

    # Visualize the results
    plt.figure()
    plt.plot(truth[:,0], truth[:,1], 'r-', label='Truth')
    plt.plot(zs[:,0],    zs[:,1],    'b+', label='Observation')
    plt.plot(xs[:,0],    xs[:,1],    'g-', label=localizer_name)
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(times, truth[:,0], 'r-', label='Truth')
    plt.plot(times, zs[:,0],    'b+', label='Observation')
    plt.plot(times, xs[:,0],    'g-', label=localizer_name)
    plt.xlabel('Time')
    plt.ylabel('X')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(times, truth[:,1], 'r-', label='Truth')
    plt.plot(times, zs[:,1],    'b+', label='Observation')
    plt.plot(times, xs[:,1],    'g-', label=localizer_name)
    plt.xlabel('Time')
    plt.ylabel('Y')
    plt.grid()
    plt.legend()

    plt.show()