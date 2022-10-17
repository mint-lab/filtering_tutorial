import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Prepare a noisy signal
    true_signal = lambda t: 10 * np.sin(2*np.pi/2*t) * np.cos(2*np.pi/10*t)
    times = np.arange(1, 10, 0.01)
    truth = true_signal(times)
    obs_signal = truth + np.random.normal(scale=1, size=truth.size)

    # Perform exponential moving average
    alpha = 0.125
    xs = []
    for z in obs_signal:
        if len(xs) == 0:
            xs.append(z)
        else:
            xs.append(xs[-1] + alpha * (z - xs[-1]))

    # Visualize the results
    plt.figure()
    plt.plot(times, obs_signal, 'b-', label='Observation', alpha=0.3)
    plt.plot(times, xs,         'g-', label='Exp. Moving Avg.')
    plt.plot(times, truth,      'r-', label='Truth', alpha=0.5)
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    plt.show()