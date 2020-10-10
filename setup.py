import setuptools

with open('README.md', 'r') as file:
    long_description = file.read()

setuptools.setup(
    name='u2net-wrapper',
    version='1.0.0',
    description='A wrapper for the U-2-net segmentation model.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/galacticglum/u2net-wrapper',
    packages=['u2net-wrapper'],
    python_requires='>=3.6',
)