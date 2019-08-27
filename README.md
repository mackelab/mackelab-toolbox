# mackelab

This repository contains Python code shared across the lab. It is public to allow functions to be used in published code.


### Installation

Install options are
  - `smt`: Install dependencies for the `Sumatra` toolbox
  - `pymc3`: Install dependencies for the `PyMC3` toolbox
  - `all` Install dependencies for all toolboxes
For the `smt` or `all` options, you will need to first install the `postgreSQL` development package:
  - `postgresql-dev` on Ubuntu
  - `postgresql-devel` on SUSE

After cloning the repository, call:
```bash
python setup.py develop --user
```

Possible setup call after navigating to location where directory was cloned:
```bash
pip install -e .[all]
```
Replace `all` with the desired install option.

### Usage

See `demo` folder for notebooks illustrating functions included in this repository.


### Contribute

- Add functions, if possible including a notebook in `demos` illustrating how/when to use the function
- Report issues

### Updating

If you installed the package using `develop`, a single `git pull` in the locally cloned folder will suffice for updating. The reason for that being that `develop` will create a symbolic link in Python's `site-packages`.

### Known issues

- Some packages are less mature than others; we need to make this more clear, but until that happens, the following lists relatively mature modules at the 'alpha' or 'beta' stage:
    + iotools (beta)
    + optimizers (alpha)
    + parameters (beta)
    + plot (beta)
    + utils (variable)
Other modules should be considered as experimental.

- Two modules have a dependency on Alex's fsGIF's project; those should be removed before being used in other projects:
  + smttk
  + sumatra