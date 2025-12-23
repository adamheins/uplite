
## Install
We use conda to ensure compatibility between pinocchio and casadi.

* Install acados and conda
* Make conda env:
  ```
  conda create -n uplite python=3.12
  conda install pinocchio -c conda-forge
  pip install -e <path_to_acados>/interfaces/acados_template
  ```

## Todo
* optimization over arm variables
  - down to jerk or acceleration
* single rigid body with contact points
  - also with no shear force
* single point with no shear acceleration
