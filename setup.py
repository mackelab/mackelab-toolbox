from setuptools import setup

setup(
    name='mackelab',
    version='0.1.0dev',
    description='Common utils for Mackelab',
    url='https://github.com/mackelab/mackelab',
    install_requires=[
        'matplotlib',
        'theano_shim >= 0.2',
        'simpleeval'
    ],
    dependency_links=[
        'git+ssh://git@github.com:alcrene/parameters.git',
    ],

    extras_require = {
        'smt': [
            'Click',
            'sumatra'
        ],
        'pymc3': [
            'pymc3'
        ],
        'all': [
            'Click', 'sumatra', 'pymc3'
        ]
    },
    entry_points = {
        'console_scripts':
            ['smttk = mackelab.smttk:cli [smt]']
    }
)
