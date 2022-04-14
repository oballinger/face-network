import streamlit as st
import pandas as pd
import numpy as np
import face_network as fn
from PIL import Image
import streamlit.components.v1 as components
from pathlib import Path
import os

tempdir=Path('tmp/')


st.title('Image Co-Appearance Network Generator')

st.write('This tool uses artificial intelligence to create a social network graph based on whether individuals appear in photographs together. This is accomplished in three steps:')

st.header('Step 1: Upload Images')


uploaded_files = st.file_uploader("Please ensure the images are either .jpg/.jpeg files. If you have lots of images, compressing the files can help speed things up.",accept_multiple_files=True, type=['jpeg','jpg'])

@st.cache
def Ingest():
	fn.overwrite(tempdir)
	for uploaded_file in uploaded_files:
		try:
			img = Image.open(uploaded_file)
			img.save(os.path.join(tempdir,uploaded_file.name))
		except:
			pass

if len(uploaded_files)>0:
	Ingest()
# Clustering 
st.header('Step 2: Cluster Faces')
st.write('This step uses a neural network to extract faces from images and cluster those that belong to the same person. You can control the face similarity threshold using the slider below, and execute clustering by pressing the button. This will generate a mosaic of faces believed to be the same person. If the clusters contain faces of multiple different people, try lowering the matching threshold and clustering again. If the same person is getting split across multiple clusters, try increasing the threshold.')

slider = st.slider(
     'Select a threshold for face similarity',
     0.0, 1.0, value=0.4)

cluster_button = st.button("Cluster")

if cluster_button:
	with st.spinner('Extracting faces...'):
		fn.extract(tempdir, exif=True)
	with st.spinner('Clustering similar faces...'):
		fn.cluster(tempdir, algorithm='chinese_whispers', initial_eps=slider, mosaic='streamlit')
	for i in range(1,11):
		try:
			st.image('tmp/Face Network/Clusters/{}.jpg'.format(i))
		except:
			pass
	pass


# Network 
st.header('Step 3: Generate a Co-Appearance Network ')
st.write('Once the clusters above look accurate, click "Generate Network". This will link together people who appear in photographs together, creating a social network graph.')

network_button = st.button("Generate Network")

if network_button:
	with st.spinner('Building Network...'):
		fn.network(tempdir)
		HtmlFile = open("tmp/Face Network/Data/Image_Network.html", 'rb')
		source_code = HtmlFile.read() 
		components.html(source_code, height=750, width=750)


st.header('Further Analysis')

netstat_button = st.button("Network Statistics")
if netstat_button:
	netstats=fn.network_analysis(tempdir)
	st.write(netstats)


#face_network_dev.cluster(photo_dir, algorithm='chinese_whispers', initial_eps=0.39, mosaic=True)
#face_network_dev.network(photo_dir)

