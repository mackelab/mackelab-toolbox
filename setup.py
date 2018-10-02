from setuptools import setup

setup(
    name='mackelab',
    version='0.1.0dev',
    description='Common utils for Mackelab',
    url='https://github.com/mackelab/mackelab',
    install_requires=[
        'matplotlib',
        'theano_shim >= 0.2',
        'simpleeval',
        'dill',
        'tqdm',
        'pandas'
    ],
    dependency_links=[
        #'git+ssh://git@github.com:alcrene/parameters.git',
        'https://github.com/alcrene/parameters'
    ],
    # use with --process-dependency_links

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
        'all': [
            'Click', 'sumatra[git]', 'pymc3', 'docutils', 'PyYAML', 'httplib2'
        ]
    },
    entry_points = {
        'console_scripts':
            ['smttk = mackelab.smttk:cli [smt]']
    }
)
