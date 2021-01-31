from setuptools import setup

setup(
    name='mackelab-toolbox',
    version='0.1.0dev3',
    description='Common utils for Mackelab',
    url='https://github.com/mackelab/mackelab_toolbox',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
    ],

    install_requires=[
        'matplotlib',
        'seaborn',
        'parameters',
        'pydantic>=1.5',
        'pyyaml',  # Required by parameters, for `import yaml`
        'simpleeval',
        'dill',
        'tqdm',
        'pandas',
        'blosc',    # Serialization typing_module.Array. Make optional ?
        'astunparse'  # Serialization of functions. Make optional ?
    ],

    packages=['mackelab_toolbox'],

    extras_require = {
        'smt': [
            'Click',
            'sumatra[git]',
            'docutils', 'PyYAML', 'httplib2',  # Required by Sumatra
	        # 'psycopg2'    # Required for using PostgreSQL Sumatra record stores
        ],
        'pymc3': [
            'pymc3'
        ],
        'theano': [
            'theano_shim>=0.2'
        ],
        'all': [
            'Click', 'sumatra[git]', 'pymc3', 'docutils', 'PyYAML', 'httplib2',# 'psycopg2',
            'theano_shim>=0.3.0dev0',
            'pygments',
            'IPython'
        ]
    },
    # entry_points = {
    #     'console_scripts':
    #         ['smttk = mackelab.smttk:cli [smt]']
    # }
)
