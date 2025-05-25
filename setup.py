from setuptools import setup, find_packages

setup(
    name="shengchengzi",
    version="0.1.0",
    author="AJ Wisniewski",
    description="Generative AI toolkit for various tasks involving Chinese characters",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="http://github.com/awisniewski21/ShengChengZi",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.10",
)