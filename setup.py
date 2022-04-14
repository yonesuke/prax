import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="prax",
    version="0.0.1",
    install_requires=[
        "jax",
        "tree"
    ],
    author="yonesuke",
    author_email="13e.e.c.13@gmail.com",
    description="JAX implementation of phase response curve",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yonesuke/prax",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)