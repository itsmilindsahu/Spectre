from setuptools import setup, find_packages

setup(
    name="spectre",
    version="0.1.0",
    description="Open-source functional-group identification from IR spectra (extending to NMR)",
    author="[Your Name], Paarth",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
    ],
    extras_require={
        "plot": ["matplotlib>=3.7"],
        "dev": ["pytest>=7.4"],
    },
    entry_points={
        "console_scripts": [
            "spectre=spectre.cli:main",
        ],
    },
)
