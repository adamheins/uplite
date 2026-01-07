<p align="center">
<img src="https://static.adamheins.com/upright/uplite.svg" alt="Lightweight robot waiter." width="15%"/>
</p>

Uplite is a modern and lightweight implementation of the core ideas from [this
paper](https://arxiv.org/abs/2305.17484) for solving the robot waiter problem.
Compared to the [original
implementation](https://github.com/utiasDSL/upright/), uplite:
* does not use ROS;
* is only written in Python;
* is simulation-only;

TODO

## Install

uplite uses [pixi](https://pixi.prefix.dev) to manage dependencies:
```sh
pixi install
```
but unfortunately acados must be installed manually:
```sh
...
pixi run pip install -e <path_to_acados>/interfaces/acados_template
```

## Todo
* single rigid body with contact points
  - also with no shear force
* single point with no shear acceleration
