import setuptools
__version__ = '0.1.0'

with open("README.md", "r") as fh:
    long_description = fh.read()
NAME='Bioplots'
try:
    f = open("requirements.txt", "rb")
    REQUIRES = [i.strip() for i in f.read().decode("utf-8").split("\n")]
    f.close()
except:
    print("'requirements.txt' not found!")
    REQUIRES = []
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
    install_requires=REQUIRES,
    python_requires='>=3, <4',
    keywords= ['Data Visualization', 'Bioinformatics','Genomics','Computational Biologist'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ]
)
