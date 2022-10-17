import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Prepare a noisy signal
    true_signal = lambda t: 10 * np.sin(2*np.pi/2*t) * np.cos(2*np.pi/10*t)
    times = np.arange(1, 10, 0.01)
    truth = true_signal(times)
    obs_signal = truth + np.random.normal(scale=1, size=truth.size)

    # Perform the simple 1-D Kalman filter
    var_init, var_q, var_r = 10, 0.02, 1
    xs, var = [], []
    for z in obs_signal:
        if len(xs) == 0:
            xs.append(z)
            var.append(var_init)
        else:
            # Predict signal change
            pred_x  = xs[-1]
            pred_var = var[-1] + var_q

            # Correct signal change
            alpha = pred_var / (pred_var + var_r)
            print(alpha)
            xs.append(pred_x + alpha * (z - pred_x))
            var.append((1 - alpha) * pred_var)

    # Visualize the results
    plt.figure()
    plt.plot(times, obs_signal, 'b-',  label='Observation', alpha=0.3)
    plt.plot(times, xs,         'g-',  label='Simple KF (x)')
    plt.plot(times, var,        'g--', label='Simple KF (Var)')
    plt.plot(times, truth,      'r-',  label='Truth', alpha=0.5)
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    plt.show()