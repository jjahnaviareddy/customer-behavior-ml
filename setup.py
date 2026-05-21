from setuptools import setup, find_packages

setup(
    name="customer-behavior-ml",
    version="1.0.0",
    description="Customer purchase intent prediction using classical ML",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/YOUR_USERNAME/customer-behavior-ml",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.2",
        "scikit-learn>=1.4",
        "xgboost>=2.0",
        "imbalanced-learn>=0.12",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "shap>=0.45",
        "pyyaml>=6.0",
        "joblib>=1.4",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
