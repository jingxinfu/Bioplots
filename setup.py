import setuptools
from Bioplots import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()
NAME='Bioplots'
setuptools.setup(
    name="Bioplots",
    version=__version__,
    author="Jingxin Fu",
    author_email="jingxinfu.tj@gmail.com",
    description="A data visualization package for bioinformatians and computational biologist",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://jingxinfu.github.io/"+NAME,
    packages=setuptools.find_packages(),
    scripts=['bin/'+NAME],
    package_data={NAME: ["data/*"],},
    include_package_data=True,
    install_requires=['pandas','numpy','matplotlib'],
    python_requires='>=2.7, <4',
    keywords= ['Data Visualization', 'Bioinformatics','Genomics','Computational Biologist'],
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ]
)
