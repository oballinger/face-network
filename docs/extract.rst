============================
Extracting Faces from Images
============================

This function extracts all faces from a directory of images using Dlib’s face detector, and must be run prior to further analysis.

.. code-block:: python
	
	face_network.extract_faces(source_dir, age_gender=False)

Parameters
==========

:source_dir: (*str*); The path to the image folder (note: this folder can contain sub-directories, so specify the highest level image directory). 
:age_gender: (*bool, defaul=False*); Estimates apparent age and gender using a pretrained Convolutional Neural Network (ResNet-50). Results are stored in the columns “age” and “gender” in the resulting dataframe. Gender is predicted on a scale from 0 (male) to 1 (female). 



Outputs
=======


This function creates a new folder called “Face Network” in your image directory. When a face is identified, it is cropped and stored in a new folder “source_dir/Face Network/Faces/”. Given “Image.jpg” containing two faces, this function will save two cropped faces: “face1_Image.jpg” and “face2_Image.jpg”. Facial encodings (128-dimensional vectors used for clustering and matching similar faces) are stored in a file called “FaceDatabase.h5”. 
 
Performance
===========

The facial extraction function is threaded, by default using n_cpus-1. Using seven cores on an M1 MacBook Pro, all 14,000 faces were extracted from the `Labeled Faces in the Wild`_ (LFW) database in 42 minutes (~6 faces per second). Facial encodings for the entire LFW database take up 15 MB (~1kb per face). 

.. _`Labeled Faces in the Wild`: http://vis-www.cs.umass.edu/lfw/
