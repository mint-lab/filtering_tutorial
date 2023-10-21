# An Intuitive Tutorial on Bayesian Filtering
## Introduction
This short tutorial aims to make readers understand Bayesian filtering intuitively. Instead of derivation of Kalman filter, I introduce Kalman filter from weighted average and moving average. I expect that readers will have intuition on Kalman filter such as meaning of equations.
* To clone this repository (codes and slides): `git clone https://github.com/mint-lab/filtering_tutorial.git`
* To install required Python packages: `pip install -r requirements.txt`
* To fork this repository to your Github: [Click here](https://github.com/mint-lab/filtering_tutorial/fork)
* To download codes and slides as a ZIP file: [Click here](https://github.com/mint-lab/filtering_tutorial/archive/master.zip)
* To see the lecture slides: [Click here](https://github.com/mint-lab/filtering_tutorial/blob/master/slides/filtering_tutorial.pdf)
* :memo: Please don't miss Roger Labbe's great book, [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)

This tutorial contains example applications to 2-D localization with various conditions. It is important for users to know how to define the following five items in their applications. I hope that readers can build their intuition from my series of examples. My codes are based on [FilterPy](https://github.com/rlabbe/filterpy/), but their contents and your understanding may not be limited to the library.
1. State variable
1. State transition function
1. State transition noise
1. Observation function
1. Observation noise



## Code Examples
### 1) From Weighted and Moving Average to Simple 1-D Kalman Filter
* Merging two weight measurements
* Merging two weight measurements with their variance ([inverse-variance weighted average](https://en.wikipedia.org/wiki/Inverse-variance_weighting))
* [1-D noisy signal filtering](https://github.com/mint-lab/filtering_tutorial/blob/master/ema_1d_signal.py) (with [exponential moving average](https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average))
* [1-D noisy signal filtering](https://github.com/mint-lab/filtering_tutorial/blob/master/ema_1d_signal.py) (with simple 1-D Kalman filter)
### 2) [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter)
* [1-D noisy signal filtering](https://github.com/mint-lab/filtering_tutorial/blob/master/kf_1d_signal.py)
  * State variable: $\mathbf{x} = x$
  * State transition function: $\mathbf{x}\_{k+1} = f(\mathbf{x}\_k; \mathbf{u}\_{k+1}) = \mathbf{x}\_k$
    * Control input: $\mathbf{u}_k = [ ]$
  * State transition noise: $\mathrm{Q} = \sigma^2_Q$
  * Observation function: $\mathbf{z} = h(\mathbf{x}) = \mathbf{x}$
    * Observation: 1-D signal value
  * Observation noise: $\mathrm{R} = \sigma^2_{R}$

* [2-D position tracking](https://github.com/mint-lab/filtering_tutorial/blob/master/kf_2d_position.py) (without class inheritance)
  * State variable: $\mathbf{x} = [x, y]^\top$
  * State transition function: $\mathbf{x}\_{k+1} = f(\mathbf{x}\_k; \mathbf{u}\_{k+1}) = \mathbf{x}\_k$
    * Control input: $\mathbf{u}_k = [ ]$
  * State transition noise: $\mathrm{Q} = \mathrm{diag}(\sigma^2_x, \sigma^2_y)$
  * Observation function: $\mathbf{z} = h(\mathbf{x}) = [x, y]^\top$
    * Observation: $\mathbf{z} = [x_{GPS}, y_{GPS}]^\top$
  * Observation noise: $\mathrm{R} = \mathrm{diag}(\sigma^2_{GPS}, \sigma^2_{GPS})$

### 3) [Extended Kalman Filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter) (EKF)
* [2-D pose tracking with simple transition noise](https://github.com/mint-lab/filtering_tutorial/blob/master/ekf_2d_pose_simple_noise.py) (without class inheritance)
  * State variable: $\mathbf{x} = [x, y, \theta, v, w]^\top$
  * State transition function: Constant velocity model (time interval: $t$)
    * Control input: $\mathbf{u}_k = [ ]$
```math
\mathbf{x}_{k+1} = f(\mathbf{x}_k; \mathbf{u}_{k+1}) = \begin{bmatrix} x_k + v_k t \cos(\theta_k + w_k t / 2) \\ y_k + v_k t \sin(\theta_k + w_k t / 2) \\ \theta_k + w_k t \\ v_k \\ w_k \end{bmatrix}
```
* * State transition noise: $\mathrm{Q} = \mathrm{diag}(\sigma^2_x, \sigma^2_y, \sigma^2_\theta, \sigma^2_v, \sigma^2_w)$ 
  * Observation function: $\mathbf{z} = h(\mathbf{x}) = [x, y]^\top$
    * Observation: $\mathbf{z} = [x_{GPS}, y_{GPS}]^\top$
  * Observation noise: $\mathrm{R} = \mathrm{diag}(\sigma^2_{GPS}, \sigma^2_{GPS})$

* [2-D pose tracking](https://github.com/mint-lab/filtering_tutorial/blob/master/ekf_2d_pose.py) (using class inheritance)
  * Its _state variable_, _state transition function_, _observation function_, and _observation noise_ are same with [the above example](https://github.com/mint-lab/filtering_tutorial/blob/master/ekf_2d_pose_simple_noise.py).
  * State transition noise
```math
\mathrm{Q} = \mathrm{W}^\top \mathrm{M} \mathrm{W} \quad \text{where} \quad \mathrm{W} = \begin{bmatrix} \frac{\partial f}{\partial v} & \frac{\partial f}{\partial w} \end{bmatrix} \quad \text{and} \quad \mathrm{M} = \begin{bmatrix} \sigma^2_v & 0 \\ 0 & \sigma^2_w \end{bmatrix}
```

* [2-D pose tracking with odometry](https://github.com/mint-lab/filtering_tutorial/blob/master/ekf_2d_pose_odometry.py)
  * Its _state transition noise_, _observation function_, and _observation noise_ are same with [the above example](https://github.com/mint-lab/filtering_tutorial/blob/master/ekf_2d_pose.py).
  * State variable: $\mathbf{x} = [x, y, \theta]^\top$
  * State transition function: Constant velocity model (time interval: $t$)
    * Control input: $\mathbf{u}_k = [v_k, w_k]^\top$
```math
\mathbf{x}_{k+1} = f(\mathbf{x}_k; \mathbf{u}_{k+1}) = \begin{bmatrix} x_k + v_{k+1} t \cos(\theta_k + w_{k+1} t / 2) \\ y_k + v_{k+1} t \sin(\theta_k + w_{k+1} t / 2) \\ \theta_k + w_{k+1} t \end{bmatrix}
```

* [2-D pose tracking with off-centered GPS](https://github.com/mint-lab/filtering_tutorial/blob/master/ekf_2d_pose_off_centered.py)
  * Its _state variable_, _state transition function_, _state transition noise_, and _observation noise_ are same with [the above example](https://github.com/mint-lab/filtering_tutorial/blob/master/ekf_2d_pose.py).
  * Observation function [[Choi20]](http://doi.org/10.1109/TITS.2019.2915108): Off-centered GPS ( $o_x$ and $o_y$ are frontal and lateral offset of the GPS.)<p/>
```math
\mathbf{z} = \begin{bmatrix} x_{GPS} \\ y_{GPS} \end{bmatrix} = h(\mathbf{x}) = \begin{bmatrix} x + o_x \cos \theta - o_y \sin \theta \\ y + o_x \sin \theta + o_y \cos \theta \end{bmatrix}
```

### 4) [Unscented Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter) (UKF)
* [2-D pose tracking with simple transition noise](https://github.com/mint-lab/filtering_tutorial/blob/master/ukf_2d_pose_simple_noise.py)
  * Its five definitions (and also implementation style) are same with [its corresponding EKF example](https://github.com/mint-lab/filtering_tutorial/blob/master/ekf_2d_pose_simple_noise.py).
* [2D pose tracking](https://github.com/mint-lab/filtering_tutorial/blob/master/ukf_2d_pose.py)
  * Its five definitions (and also implementation style) are same with [its corresponding EKF example](https://github.com/mint-lab/filtering_tutorial/blob/master/ekf_2d_pose.py).

### 5) [Particle Filter](https://en.wikipedia.org/wiki/Particle_filter)
* [2-D pose tracking](https://github.com/mint-lab/filtering_tutorial/blob/master/pf_2d_pose.py)
  * Its five definitions (and also implementation style) are same with [its corresponding EKF example](https://github.com/mint-lab/filtering_tutorial/blob/master/ekf_2d_pose.py).



## References
* [FilterPy Documentation](https://filterpy.readthedocs.io/en/latest/): FilterPy API and examples
  * Bookmarks: [Kalman filter](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html), [EKF](https://filterpy.readthedocs.io/en/latest/kalman/ExtendedKalmanFilter.html), [UKF](https://filterpy.readthedocs.io/en/latest/kalman/UnscentedKalmanFilter.html), [resampling](https://filterpy.readthedocs.io/en/latest/monte_carlo/resampling.html) (for particle filter)
* [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python): A great introduction on Bayesian filtering with FilterPy
  * Bookmarks: [Table of Contents](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb), [Kalman filter](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/08-Designing-Kalman-Filters.ipynb), [EKF](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/11-Extended-Kalman-Filters.ipynb), [UKF](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb), [particle filter](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb)
* Particle filter
  * [pfilter](https://github.com/johnhw/pfilter), John Williamson
  * [Monte Carlo Particle Filter for Localization](https://github.com/p16i/particle-filter), Pattarawat Chormai
  * [particle_filter_demo](https://github.com/mjl/particle_filter_demo), Martin J. Laubach
  * [Particle Filter for _Turtle_ Localization](https://github.com/leimao/Particle-Filter), Lei Mao



## Authors
* [Sunglok Choi](https://mint-lab.github.io/sunglok/)
* [Hyunkil Hwang](https://github.com/Hyunkil76)



## Acknowledgement
This tutorial was supported by the following R&D projects in Korea.
*  AI-based Localization and Path Planning on 3D Building Surfaces (granted by [MSIT](https://www.msit.go.kr/)/[NRF](https://www.nrf.re.kr/), grant number: 2021M3C1C3096810)
