from setuptools import setup

setup(
    name="texture_gen",
    version="0.0.0",
    author="Xiangyi Gao",
    description="generate texture form an image.",
    packages=["texture_gen"],
    install_requires=[
        "torchvision==0.10.1",
        "scikit-learn==0.24.2",
        "pytorch-lightning==1.4.7",
    ],
)
