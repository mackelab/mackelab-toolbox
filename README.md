# mackelab

#### The lab wiki is associated with this repo, see: https://github.com/mackelab/mackelab/wiki


## Shared code

In addition, this repository contains Python code shared across the lab.


### Installation

After cloning the repository, call:
```bash
python setup.py develop --user
```

Possible setup call after navigating to location where directory was cloned:
```bash
pip install -e .[all]
```
Possible install options are
  - `smt`: Install dependencies for the `Sumatra` toolbox
  - `pymc3`: Install dependencies for the `PyMC3` toolbox
  - `all` Install dependencies for all toolboxes

### Usage

See `demo` folder for notebooks illustrating functions included in this repository.


### Contribute

- Add functions, if possible including a notebook in `demos` illustrating how/when to use the function
- Report issues

### Updating

If you installed the package using `develop`, a single `git pull` in the locally cloned folder will suffice for updating. The reason for that being that `develop` will create a symbolic link in Python's `site-packages`.
