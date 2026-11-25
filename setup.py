import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Author packages (HISTONAUT, COSMIX, MILKYWAY) are not on PyPI: install them
# from source first (`pip install -e <path>`), pip then finds them installed.

setuptools.setup(
    name="proifhesi",
    version="0.0.1",
    author="mblons",
    author_email="matthieu.blons@hotmail.fr",
    description="Proteomic signature from HE stained WSI.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    # Only the LoRA fine-tuning / validation code is importable; preprocessing/
    # and downstream/ are scripts, run with `python -m` from the repo root.
    packages=setuptools.find_packages(
        include=[
            "finetuning",
            "finetuning.*",
            "validation",
            "validation.*",
        ]
    ),
    install_requires=[
        "histonaut",
        "torch>=2.5",
        "torchvision>=0.20",
        "peft>=0.15",
        # GaussNoise(var_limit=...) API, changed in later releases
        "albumentations==1.4.21",
        "torchinfo",
        "numpy<2",
        "pandas>=2.2",
        "h5py>=3.13",
        "scikit-learn>=1.5",
        "Pillow",
        "pyyaml",
        "tqdm",
        "wandb",
        "tensorboard",
    ],
    extras_require={
        "preprocessing": [
            "cosmix",
            "pyvips",
            "opencv-python",
            "matplotlib",
            "joblib",
        ],
        "downstream": [
            "milkyway",
            "scikit-survival>=0.23",
            "xgboost>=3.0",
            "scipy",
            "joblib",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
