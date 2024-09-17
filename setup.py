#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='ixdrl',
    version='1.0',
    description='IxDRL: Explainable Deep Reinforcement Learning Toolkit based on Analyses of Interestingness',
    url='https://github.com/SRI-AIC/IxDRL',
    author='Pedro Sequeira',
    author_email='pedro.sequeira@sri.com',
    packages=find_packages(),
    scripts=[
    ],
    install_requires=[
        'findiff',
        'hyperopt',
        'joblib',
        'kaleido',
        'numpy==1.22.0',
        'matplotlib==3.4.2',
        'opencv-python==4.10.0.84',
        'pandas==1.4.1',
        'plotly',
        'protobuf==3.20.3',
        'pymongo==3.13.0',
        'scipy>=1.9.0',
        'scikit-video',
        'setuptools<=69.5.1',
        'shap==0.41.0',
        'tqdm',
        'xgboost==1.6.0',
    ],
    extras_require={
        'rllib': [
            'ray[default,rllib,tune]==2.9.2',
            'gymnasium[atari,accept-rom-license,classic-control]==0.28.1',
            'moviepy'
        ],
        'gui': [
            'altair==4.2.0',
            'streamlit==1.3.1',
            'streamlit_player',
            'streamlit_autorefresh',
            'streamlit_plotly_events',
            'watchdog',
        ]
    },
    zip_safe=True
)
