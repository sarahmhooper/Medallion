from setuptools import find_packages, setup

setup(
    name="Medallion",
    version="0.0.1",
    description=(
        "Training segmentation models with less labeled data."
    ),
    install_requires=[
        'emmental @ git+https://github.com/senwu/emmental@sen/online_score',
        "torch==1.13.1",
        "torchvision==0.10.0",
        "matplotlib>=3.3.4",
        "scikit-image>=0.17.2",
        "seaborn>=0.11.1",
        "urllib3==1.26.12",
        "setuptools==70.0.0"
    ],
    scripts=["bin/image_segmentation"],
    packages=find_packages(),
)
