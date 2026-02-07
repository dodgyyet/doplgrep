from setuptools import setup, find_packages

setup(
    name="doplgrep",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "datasets>=2.14.0",
        "tqdm>=4.66.0",
        "accelerate>=0.24.0",
    ],
    entry_points={
        "console_scripts": [
            "doplgrep=doplgrep.__main__:main",
        ],
    },
    python_requires=">=3.8",
    author="Joseph Roche",
    description="Face doppelganger search using DINOv3 embeddings",
)