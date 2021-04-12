# Mackelab toolbox

This repository contains Python code shared across the lab. It is public to allow functions to be used in published code.


## Installation

### Contexts

It is important to note that **all dependencies are optional**. Being a collaborative toolbox, this package contains a relatively large number of functions for many different uses, extending or patching the functionality of a large number of sometimes heavy packages.
For example, there are very few situations where it would make sense to simultaneously install *PyTorch*, *TensorFlow* and *JaX*, but helper functions for any of these could be included in this toolbox.

Thus, to avoid forcing users to install unneeded dependencies, modules are organized into different *contexts*. When installing the toolbox, one specifies which contexts are needed as optional or ['extra' dependencies](https://setuptools.readthedocs.io/en/latest/userguide/dependency_management.html#optional-dependencies). The usual `'all'` option is provided to install all dependencies, which is especially useful for running the test suite.

Each context is associated to one or more modules. To be able to import a particular module, simply install a context associated with it.

These are the currently defined contexts, and their associated modules:
  - `'iotools'`: [*iotools.py*](mackelab_toolbox/iotools.py)
  - `'pymc3'`: [*pymc3.py*](mackelab_toolbox/pymc3.py), [*pymc_typing.py*](mackelab_toolbox/pymc_typing.py), + everything from `'theano'`, `'typing'`
  - `'theano'`: [*theano.py*](mackelab_toolbox/theano.py), [*cgshim.py*](mackelab_toolbox.py), [*optimizers.py*](mackelab_toolbox/optimizers.py), + everything from `'typing'`
  - `'torch'`: [*torch.py*](mackelab_toolbox/torch.py), + everything from `'iotools'`, `'typing'`
  - `'tqdm'`: [*tqdm.py*](mackelab_toolbox/tqdm.py)
  - `'typing'`: [*typing.py*](mackelab_toolbox/typing_module.py), [*units.py*](mackelab_toolbox/units.py), [*pydantic.py*](mackelab_toolbox/pydantic.py), [*serialize.py*](mackelab_toolbox/serialize.py), [*transform.py*](mackelab_toolbox/transform.py)
  - `'utils'`: [*utils.py*](mackelab_toolbox/utils.py), [*meta.py*](mackelab_toolbox/meta.py)
  - `'all'`: *everything*

### User installation

If you only need *mackelab_toolbox* as a dependency, the following will suffice

> Eventually this package will be placed on PyPI, at which point the usual `pip install mackelab_toolbox[…]` will work.

```bash
pip install --upgrade pip wheel
pip install "mackelab-toolbox[…] @ git+https://github.com/mackelab/mackelab-toolbox#egg=mackelab-toolbox"
```

For example, you need functions from modules associated to the `'typing'` and `'utils'` contexts, you would do the following:

```bash
pip install --upgrade pip wheel
pip install "mackelab-toolbox[typing,utils] @ git+https://github.com/mackelab/mackelab-toolbox#egg=mackelab-toolbox"
```

Alternatively, you can clone the repository first and then install as usual:
```bash
pip install --upgrade pip wheel
git clone https://github.com/mackelab/mackelab-toolbox.git
pip install mackelab-toolbox[typing,utils]
```

As usual, it is recommended to install within a virtual environment.

> Upgrading _pip_ and _wheel_ is not usually required, but in some cases can avoid problems. It also prevents pip from warning that it is out of date.

> If the toolbox functionality you need is still in development, you will likely find that cloning locally and performing an editable install (with `pip install -e`) is easier to keep up to date. See [Contributing](#Contributing) below.

> **Caveat for the _theano_ context**: The *theano_shim* package is not yet on pip, and therefore must be installed from the repo before installing *mackelab-toolbox*. This is only required if you install the _theano_ context.

### Specifying as a dependency

Due to its nature, required features in this package will often be developed as they are needed in some other project. Thus it is not always possible to simply install the version from _PyPI_. If you want your project to install _mackelab_toolbox_  from GitHub, you can add the following to its [requirements](https://pip.pypa.io/en/stable/user_guide/#requirements-files) files:

```
-e git+https://github.com/mackelab/mackelab-toolbox@master#egg=mackelab-toolbox[typing,utils]
```

This allows you to specify the branch (here `master`) and the required contexts (here `typing` and `utils`).

### Usage

See `demo` folder for notebooks illustrating functions included in this repository.

> (TODO) We should use JupyterBook to convert demo & source notebooks into searchable documentation.

The recommended import abbreviation is `mtb`:

```python
import mackelab_toolbox as mtb
import mackelab_toolbox.iotools
from mackelab_toolbox.typing import Array

mtb.iotools.save(...)

def f(x: Array):
  ...
```

### Updating

If you cloned the repository and installed with `-e`, then a single `git pull` in the locally cloned folder will suffice for updating.

Otherwise, use
```bash
pip install --update "mackelab-toolbox[…] @ git+https://github.com/mackelab/mackelab-toolbox#egg=mackelab-toolbox"
```

### Contributing

If you intend to make modifications to *mackelab-toolbox*, clone to the location where you keep your working files, and install from there. Use a development install (`-e`) to avoid having to reinstall each time you make a modification:

```bash
git clone https://github.com/mackelab/mackelab-toolbox.git
pip install -e mackelab-toolbox[all]
```

As above, use options to indicate which contexts you need.

#### Guidelines

- When adding functions, consider using JupyText to include rich inline documentation in the source file.
  > **Tip:** You can use `if __name__ == "__main__"` guards to include examples directly in source files.
- If possible, also include a notebook in `demos` illustrating how/when to use the function.
- Report issues.

#### Running tests

You will need _pytest_ to run the test suite:

```bash
pip install pytest pytest-forked
```

Then run tests with

```bash
pytest --forked
```

The *pytest-forked* package (which provides the `--forked` option) are required due to how the *typing* context implements dynamic types. This may change in the future.

> At present, it is difficult to run tests without having all dependencies installed. This is something we may improve in the future.

### Known issues

- By its nature, this toolbox package will contain modules of varying levels of maturity. It is not clear what the best way to indicate maturity level would be – probably an indication at the the function would be needed.
  + At present (Apr 2021), all modules are somewhere between an 'alpha' and a 'beta' stage: they have all been use-tested, but may still see structural modifications. In particular:
    + Much of the functionality in *plot.py* is probably redundant with *Seaborn*.
    + The requirement to “freeze” types from *typing.py* is a confusing hack and it would be great if we could do without.
