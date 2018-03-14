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
        'tqdm'
    ],
    dependency_links=[
        'git+ssh://git@github.com:alcrene/parameters.git',
    ],

    extras_require = {
        'smt': [
            'Click',
            'sumatra[git]',
            'docutils', 'PyYAML', 'httplib2'  # Required by Sumatra
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
