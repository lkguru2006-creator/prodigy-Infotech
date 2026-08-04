from setuptools import find_packages, setup

setup(
    name="house-price-predictor",
    version="1.0.0",
    description="Enterprise-grade linear regression pipeline for house price prediction.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
)
