Face-Network Documentation
========================================
|PyPI badge|

.. |PyPI badge| image:: https://badge.fury.io/py/face-network.svg
   :target: https://badge.fury.io/py/face-network


Given a large volume of images of people, this tool uses artificial intelligence to generate an interactive social network graph based on co-appearance in images: 

.. raw:: html
   
   <iframe src="_static/Image_Network.html" scrolling="no" height="600px" width="100%" style="border:1px solid black;"></iframe>

Each node is a person, and each edge linking two nodes represents an image in which the two people appear together. The network above can be clicked, dragged, and zoomed to reveal the latent social structure present in unstructured image data. 

A folder of images can be turned into a co-appearance network in the following three steps:

.. toctree::
   :maxdepth: 1

   extract
   cluster
   network

The diagram below illustrates this process: 

.. figure::  images/image3.png

