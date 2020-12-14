import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyimgutils",
    version="0.0.1",
    author="C. Gus Becker",
    author_email="cgusbecker@gmail.com",
    description="A package containing general image processing modules.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cgusb/pyimgutils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "matplotlib",
        "skimage"
    ]
)