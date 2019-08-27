# mackelab

This repository contains Python code shared across the lab. It is public to allow functions to be used in published code.


### Installation

After cloning the repository, call:
```bash
python setup.py develop --user
```

### Usage

See `demo` folder for notebooks illustrating functions included in this repository.


### Contribute

- Add functions, if possible including a notebook in `demos` illustrating how/when to use the function
- Report issues

### Updating

If you installed the package using `develop`, a single `git pull` in the locally cloned folder will suffice for updating. The reason for that being that `develop` will create a symbolic link in Python's `site-packages`.
