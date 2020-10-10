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
    packages=['u2net_wrapper'],
    install_requires=[
        'requests==2.24.0',
        'clint==0.5.1',
        'scikit-image==0.17.2'
    ],
    python_requires='>=3.6',
)