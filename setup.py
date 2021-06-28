from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

# Setting up
setup(
    name="face-network",
    version='1.0.2',
    author="Ollie Ballinger",
    author_email="ollie.l.ballinger@gmail.com",
    license="MIT",
    url="https://github.com/oballinger/face-network",
    description="Create a social network graph based on coappearance in images",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['opencv-python','dlib','face_recognition','numpy','pandas','pathlib','joblib','omegaconf','sklearn', 'tensorflow', 'cmake'],
    keywords=['python', 'image', 'network', 'AI', 'neural network'],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ]
)

# This call to setup() does all the work
'''
setup(
    packages=["face-network"],
    include_package_data=True,
,
)
'''