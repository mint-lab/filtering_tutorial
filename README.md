### Code Examples
* :pencil: The following five definitions are important to design and analyze Bayesian filtering. Please refer my examples in the view of the five definitions, their implementation, and results.

* **Kalman filter** [[Wikipedia]](https://en.wikipedia.org/wiki/Kalman_filter)
  * [2D position estimation](https://github.com/mint-lab/filtering_tutorial/blob/master/kf_2d_position.py)
    * State variable: $\mathbf{x} = [x, y]^\top$
    * State transition function: $\mathbf{x}_{t+1} = f(\mathbf{x}_t) = \mathbf{x}_t$
    * State transition noise: $Q = \mathrm{diag}(\sigma^2_x, \sigma^2_y)$
    * Observation function: $\mathbf{z} = h(\mathbf{x}) = [x, y]^\top$ (similar to GPS)
    * Observation noise: $R = \mathrm{diag}(\sigma^2_{GPS}, \sigma^2_{GPS})$
  
* **EKF** [[Wikipedia]](https://en.wikipedia.org/wiki/Extended_Kalman_filter)
  * [2D pose estimation with simple transition noise](https://github.com/mint-lab/filtering_tutorial/blob/master/ekf_2d_pose_simple_noise.py)
    * State variable: $\mathbf{x} = [x, y, \theta, v, w]^\top$
    * State transition function: Constant velocity model ($\Delta$: time interval)
      $$\mathbf{x}_{t+1} = f(\mathbf{x}_t) = \begin{bmatrix} x_t + v_t \Delta \cos(\theta + w_t \Delta / 2) \\\ y_t + v_t \Delta \sin(\theta + w_t \Delta / 2) \\\ \theta_t + w_t \Delta \\\ v_t \\\ w_t \end{bmatrix}$$
    * State transition noise: $Q = \mathrm{diag}(\sigma^2_x, \sigma^2_y, \sigma^2_\theta, \sigma^2_v, \sigma^2_w)$ 
    * Observation function: $\mathbf{z} = h(\mathbf{x}) = [x, y]^\top$ (similar to GPS)
    * Observation noise: $R = \mathrm{diag}(\sigma^2_{GPS}, \sigma^2_{GPS})$
  * [2D pose estimation](https://github.com/mint-lab/filtering_tutorial/blob/master/ekf_2d_pose.py)
    * Its _state variable_, _state transition function_, _observation function_, and _observation noise_ are same with the above example.
    * State transition noise: $Q = W^\top M W$ where $W = \begin{bmatrix} \frac{\partial f}{\partial v} & \frac{\partial f}{\partial w}\end{bmatrix}$ and $M = \begin{bmatrix} \sigma^2_v & 0 \\\ 0 & \sigma^2_w \end{bmatrix}$
  * [2D pose estimation with off-centered GPS](https://github.com/mint-lab/filtering_tutorial/blob/master/ekf_2d_pose_off_centered.py)
    * Its _state variable_, _state transition function_, _state transition noise_, and _observation noise_ are same with the above example.
    * Observation function [[Choi20]](http://doi.org/10.1109/TITS.2019.2915108): Off-centered GPS ($o_x$ and $o_y$ are frontal and lateral offset of the GPS.)
      $$\mathbf{z} = h(\mathbf{x}) = \begin{bmatrix} x + o_x \cos \theta - o_y \sin \theta \\\ y + o_x \sin \theta + o_y \cos \theta \end{bmatrix}$$
  
* **UKF** [[Wikipedia]](https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter)
  * [2D pose estimation with simple transition noise](https://github.com/mint-lab/filtering_tutorial/blob/master/ukf_2d_pose_simple_noise.py) (similar its [EKF example](https://github.com/mint-lab/filtering_tutorial/blob/master/ekf_2d_pose_simple_noise.py))
  * [2D pose estimation](https://github.com/mint-lab/filtering_tutorial/blob/master/ukf_2d_pose.py) (similar to its [EKF example](https://github.com/mint-lab/filtering_tutorial/blob/master/ekf_2d_pose.py))

* **Particle filter** [[Wikipedia]](https://en.wikipedia.org/wiki/Particle_filter)
  * [2D pose estimation](https://github.com/mint-lab/filtering_tutorial/blob/master/ukf_2d_pose.py) (similar to its [EKF example](https://github.com/mint-lab/filtering_tutorial/blob/master/pf_2d_pose.py))


### References
* [FilterPy Documentation](https://filterpy.readthedocs.io/en/latest/): FilterPy API and examples
  * Useful parts: [Kalman filter](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html), [EKF](https://filterpy.readthedocs.io/en/latest/kalman/ExtendedKalmanFilter.html), [UKF](https://filterpy.readthedocs.io/en/latest/kalman/UnscentedKalmanFilter.html)
* [Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python): A great introduction on Bayesian filtering with FilterPy
  * Useful parts: [Table of Contents](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/table_of_contents.ipynb), [EKF](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/11-Extended-Kalman-Filters.ipynb), [UKF](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb), [particle filter](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb)
* Particle filter
  * [pfilter](https://github.com/johnhw/pfilter), John Williamson
  * [Monte Carlo Particle Filter for Localization](https://github.com/p16i/particle-filter), Pattarawat Chormai
  * [particle_filter_demo](https://github.com/mjl/particle_filter_demo), Martin J. Laubach
  * [Particle Filter for _Turtle_ Localization](https://github.com/leimao/Particle-Filter), Lei Mao
