from setuptools import setup, find_packages

setup(
    name='mackelab-toolbox',
    python_requires='>=3.6',
    version='0.2.0',
    description='A set of common Python utilities for computational science',
    url='https://github.com/mackelab/mackelab_toolbox',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
    ],

    # Since this is a collaborative toolbox, it contains a relatively large
    # number of functions for many different uses, extending or patching the
    # functionality of a large number of packages.
    # Thus it has a lot of dependencies.
    # However, since each user is likely to need only a small subset, we don’t
    # want to force the installaton of all dependencies.
    # SOLUTION: All dependencies are specified as 'extras', so one can install
    # based on what the toolbox will be used for:
    # >>> pip install[iotools, plotting]
    # The usual 'all' extra is the union of all contexts except 'dev'.
    install_requires=[
    ],

    packages=find_packages(),

    extras_require = {
        'iotools': [
            'numpy',
            'dill',   # Should only include in 'all'; see TODO in iotools.py
            'parameters'
        ],
        'parameters': [
            'parameters',
            'pyyaml',  # Required by parameters, for `import yaml`
            'tqdm',
            'pandas'
        ],
        'plotting': [
            'IPython',
            'matplotlib',
            #'seaborn',  # Not currently used in plot.py
        ],
        'viz': [
            'matplotlib',
            'seaborn',
            'holoviews',
            'tabulate'
        ],
        'pymc3': [
            'pymc3',
            'matplotlib',
            'seaborn'
            # Everything from theano & typing
            'theano_shim>=0.2',
            'pydantic>=1.8',
            'blosc',
            'astunparse',
            'simpleeval',
        ],
        'stats': [
            'scipy',
            'numpy',
            # Everything from typing
            'pydantic>=1.8',
            'blosc',    # Serialization typing_module.Array.
            'astunparse',  # Serialization of functions.
            'simpleeval',  # Serialization of transforms (TransformedVar)
        ],
        'test': [
            'pytest',
            'pytest-forked'  # Required because 'typing.freeze_types' can only be called once
        ],
        'theano': [
            'theano_shim>=0.2',
            # Everything from typing
            'pydantic>=1.5',
            'blosc',
            'astunparse',
            'simpleeval',
        ],
        'torch': [
            'torch',
            # Everything from 'iotools'
            'numpy',
            'dill',   # Should only include in 'all'; see TODO in iotools.py
            'parameters'
            # Everything from 'typing'
            'pydantic>=1.5',
            'blosc',
            'astunparse',
            'simpleeval',
        ],
        'tqdm': [
            'tqdm'
        ],
        'typing': [  # typing & serialization
            'pydantic>=1.8',
            'blosc',    # Serialization typing_module.Array.
            'astunparse',  # Serialization of functions.
            'simpleeval',  # Serialization of transforms (TransformedVar)
        ],
        'utils': [
            'IPython',
            'pygments',
            'termcolor',  # Optional dependency for TimeThis
            'simpleeval'  # Dependency for total_size_handler
        ],
        'dev': [  # Extra dependencies only required for docs; NOT included in 'all'
        ],
        'all': [  # Does not include 'dev' packages
            'numpy', 'dill',
            'parameters', 'pyyaml', 'tqdm', 'pandas',
            'IPython', 'matplotlib',
            'tabulate', 'holoviews',
            'pymc3', 'seaborn', 'theano_shim>=0.3.0dev0',
            'pydantic>=1.8', 'blosc', 'astunparse', 'simpleeval',
            'pytest', 'pytest-forked',
            'torch',
            'tqdm',
            'pygments',
            # Not (yet) in any context, but some functionality for them
            'pint', 'quantities'
        ]
    },
)
