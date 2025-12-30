
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
