from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A modular, class-based machine learning pipeline for predicting customer churn."

__version__ = "0.1.0" 

REPO_NAME = "Customer-churn-prediction"
AUTHOR_USER_NAME = "rezjsh"
SRC_REPO = "customer_churn" 
AUTHOR_EMAIL = "your.email@example.com"  

# Core dependencies required for the basic framework to function
INSTALL_REQUIRES = [
    "pandas>=1.3.0",
    "numpy>=1.21.0",
    "scikit-learn>=1.0.0",
    "xgboost>=1.5.0",
    "matplotlib>=3.4.0",
    "seaborn>=0.11.0",
    "shap>=0.40.0"
]

# Optional dependencies for development and interactive environments
EXTRAS_REQUIRE = {
    "dev": ["pytest>=7.0.0", "black", "flake8"],
    "notebooks": ["jupyter", "ipykernel"]
}

setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A modular, class-based machine learning pipeline for Telco customer churn prediction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    
    # Package Discovery
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # Dependencies
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Metadata & Entry Points
    python_requires=">=3.12.4",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # Allows you to run the pipeline directly via command line
    entry_points={
        "console_scripts": [
            "run_pipeline=customer_churn.pipeline:main", 
        ]
    },
)