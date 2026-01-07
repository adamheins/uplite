<p align="center">
<img src="https://static.adamheins.com/upright/uplite.svg" alt="Lightweight robot waiter." width="15%"/>
</p>

Uplite is a modern and lightweight implementation of the core ideas from [this
paper](https://arxiv.org/abs/2305.17484) (see full citation below) for solving
the robot waiter problem. Compared to the [original
implementation](https://github.com/utiasDSL/upright/), uplite currently:
* does not use ROS;
* is only written in Python;
* is simulation-only;
* only supports one balanced object;
* does not implement obstacle avoidance;
* generates and tracks an offline plan rather than online MPC;
* does not include full (naive) friction constraints.

This lack of extra features allows the codebase to remain a simple starting
point for those wishing to demonstrate or research the waiter's problem.
However, the code should also be easy to modify to add features and extend for
your usecase.

## Install

uplite uses [pixi](https://pixi.prefix.dev) (an alternative to conda) to manage
dependencies. Assuming you have pixi installed, do:
```sh
# change destination paths as you see fit
git clone https://github.com/adamheins/uplite ~/uplite
cd ~/uplite
pixi install
```
Unfortunately, there is no conda package for
[acados](https://docs.acados.org/), so it must be installed
manually:
```sh
# see also <https://docs.acados.org/installation/index.html>
# ensure you have make and cmake on your system
git clone https://github.com/acados/acados.git ~/acados
cd ~/acados
git submodule update --recursive --init
mkdir -p build
cd build
cmake ..
make install -j4

# once acados is built, install the Python interface into your uplite pixi
# workspace
cd ~/uplite
pixi run pip install -e <path_to_acados>/interfaces/acados_template
```

## Todo
* single rigid body with contact points
  - also with no shear force
* single point with no shear acceleration


## Citations

If you find this work useful, feel free to cite the [original
paper](https://doi.org/10.1109/LRA.2023.3324520):
```
@article{heins2023upright,
  title = {Keep It Upright: Model Predictive Control for Nonprehensile Object Transportation With Obstacle Avoidance on a Mobile Manipulator},
  author = {Adam Heins and Angela P. Schoellig},
  journal = {{IEEE Robotics and Automation Letters}},
  number = {12},
  volume = {8},
  pages = {7986--7993},
  doi = {10.1109/LRA.2023.3324520},
  year = {2023},
}
```
