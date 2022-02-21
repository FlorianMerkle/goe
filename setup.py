import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='goe',
    version='0.0.0',
    author='MCI-DIBSE',
    author_email='florian.merkle@mci.edu',
    description='Code frequently used for the Game Over Eva(sion) project',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/FlorianMerkle/goe-code',
    packages=setuptools.find_packages(),
    install_requires=[
        'torch',
        'matplotlib',
        'pandas',
        'numpy',
        'torchvision',
    ],
)
