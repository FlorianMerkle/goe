import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='goe-code',
    version='0.0.0',
    author='MCI-DIBSE',
    author_email='florian.merkle@mci.edu',
    description='Code frequently used for the Game Over Eva(sion) project',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mike-huls/toolbox',
    project_urls = {
        "Bug Tracker": "https://github.com/mike-huls/toolbox/issues"
    },
    license='MIT',
    packages=['toolbox'],
    install_requires=['requests'],
)