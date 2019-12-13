from setuptools import setup

setup(
    name='mackelab-toolbox',
    version='0.1.0dev',
    description='Common utils for Mackelab',
    url='https://github.com/mackelab/mackelab_toolbox',
    install_requires=[
        'matplotlib',
        'seaborn',
        'parameters',
        'pyyaml',  # Required by parameters, for `import yaml`
        'simpleeval',
        'dill',
        'tqdm',
        'pandas'
    ],

    packages=['mackelab_toolbox'],

    extras_require = {
        'smt': [
            'Click',
            'sumatra[git]',
            'docutils', 'PyYAML', 'httplib2',  # Required by Sumatra
	        'psycopg2'    # Required for using PostgreSQL Sumatra record stores
        ],
        'pymc3': [
            'pymc3'
        ],
        'luigi': [
            'psutil'
        ],
        'theano': [
            'theano_shim>=0.2'
        ],
        'all': [
            'Click', 'sumatra[git]', 'pymc3', 'docutils', 'PyYAML', 'httplib2', 'psycopg2',
            'psutil',
            'theano_shim>=0.2'
        ]
    },
    # entry_points = {
    #     'console_scripts':
    #         ['smttk = mackelab.smttk:cli [smt]']
    # }
)
