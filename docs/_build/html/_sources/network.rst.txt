============================
Creating a Network
============================

Having identified individuals across multiple pictures, this function generates a force directed graph based on co-appearance in images. Each individual is a node, and each co-appearance is an edge. 

.. code-block:: python

	face_network.network(photo_dir, scale=10)


Parameters
==========

:source_dir: (*str*) The path to the image folder 
:scale: (*int, default=10*) Dictates the size of the nodes

Outputs
=======

A file called “Image_Network.html” is created in “photo_directory/Face Network/Data/”.The graph can be opened in a web browser and is fully interactive. Hovering over a node will display a tooltip showing the cluster’s unique identifier. This corresponds to the filenames of the mosaics generated in the previous step.


.. raw:: html
   
   <iframe src="_static/Image_Network.html" scrolling="no" height="600px" width="100%" style="border:1px solid black;"></iframe>



