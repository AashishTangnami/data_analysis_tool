"""
Setup script for Dynamic Data Analysis Platform.
"""
from setuptools import setup, find_packages

setup(
    name="dynamic_data_analysis_platform",
    version="0.1.0",
    description="A platform for automated data analysis with descriptive, diagnostic, predictive, and prescriptive capabilities",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
    ],
    entry_points={
        "console_scripts": [
            "ddap=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
