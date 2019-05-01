import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sghcm_gy-ywgej9",
    version="0.0.1",
    author="Yiwei Gong, Chao Yang",
    author_email="yiwei.gong@duke.edu",
    description="Stochastic Gradient Hamiltonian Monte Carlo",
    url="https://github.com/ywgej9/STA663-Final-Project.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
	zip_safe=True
)