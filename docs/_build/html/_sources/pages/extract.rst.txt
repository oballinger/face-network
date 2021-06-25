============================
Extracting Faces from Images
============================

```python
face_network.extract_faces(source_dir, age_gender=False)
```

This function extracts all faces from a directory of images using Dlib’s face detector, and must be run prior to further analysis.

### Outputs: 

This function creates a new folder called “Face Network” in your image directory. When a face is identified, it is cropped and stored in a new folder “source_dir/Face Network/Faces/”. Given “Image.jpg” containing two faces, this function will save two cropped faces: “face1_Image.jpg” and “face2_Image.jpg”. Facial encodings (128-dimensional vectors used for clustering and matching similar faces) are stored in a file called “FaceDatabase.h5”. 
