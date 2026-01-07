<p align="center">
<img src="https://static.adamheins.com/upright/uplite.svg" alt="Lightweight robot waiter." width="15%"/>
</p>

Uplite is a modern and lightweight implementation of the core ideas from [this
paper](https://arxiv.org/abs/2305.17484) (see full citation below) for solving
the robot waiter problem. Compared to the [original
implementation](https://github.com/utiasDSL/upright/), uplite currently:
* does not use ROS;
* is written in Python only (underlying libraries of course use C/C++, but the
  user doesn't have to);
* is simulation-only;
* uses a fixed-base arm rather than a mobile manipulator;
* only supports a single transported object (a box);
* does not implement obstacle avoidance;
* generates and tracks an offline plan rather than online MPC;
* does not include full (naive) friction constraints;
* is **vastly easier** to set up, install, and modify.

This lack of extra features allows the codebase to remain a simple starting
point for those wishing to work on or play with the waiter's problem. However,
the code should also be easy to modify to add features and extend for your
use-case.

## Install

Uplite uses [pixi](https://pixi.prefix.dev) (an alternative to conda) to manage
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
pixi run pip install -e ~/acados/interfaces/acados_template
```

## Usage

A PyBullet simulation of a fixed-based arm moving a box on a tray is provided
in `scripts/main.py`. The constraint type can be passed as a command line
argument using the `-c` or `--constraint` option; e.g.,
```
# use the 'robust' constraint proposed in the paper
# other possible constraint types are 'none', 'upward', and 'aligned'
pixi run scripts/main.py -c robust
```
This example is currently set up to fail (i.e., the box is dropped) with all
constraint types except for `robust`. Feel free to change the values of the
constants at the top of the script to obtain different behaviours.

The constraint types `none`, `upward`, and `robust` correspond to those
described in the [original paper](https://arxiv.org/abs/2305.17484). The `full`
constraints are not currently implemented. The `aligned` constraints (which
simply force the tray's normal to be aligned with the total acceleration
vector) were not included in the original paper but are described in Section
4.8.3 of [my thesis](https://static.adamheins.com/thesis.pdf).

To modify additional parameters of the trajectory optimization problem, edit
the file `uplite/planner.py`.

## Citation

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

## License

MIT (see the LICENSE file).
