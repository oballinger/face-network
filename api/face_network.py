import os
#import cv2
import json
import dlib
import shutil
import joblib
import hashlib
import exifread
import warnings
import numpy as np
import pandas as pd
import face_recognition
from pathlib import Path
from joblib import Parallel, delayed

#warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore")

def get_model(cfg):
    from tensorflow.keras import applications
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import SGD, Adam

    base_model = getattr(applications, cfg.model.model_name)(
        include_top=False,
        input_shape=(cfg.model.img_size, cfg.model.img_size, 3),
        pooling="avg"
    )
    features = base_model.output
    pred_gender = Dense(units=2, activation="softmax", name="pred_gender")(features)
    pred_age = Dense(units=101, activation="softmax", name="pred_age")(features)
    
    model = Model(inputs=base_model.input, outputs=[pred_gender, pred_age])
    return model

def overwrite(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)


from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file

detector = dlib.get_frontal_face_detector()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = '6d7f7b7ced093a8b3ef6399163da6ece'

weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="pretrained_models",
                       file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

# load model and weights
model_name, img_size = Path(weight_file).stem.split("_")[:2]

img_size = int(img_size)
cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])

model = get_model(cfg)
model.load_weights(weight_file)


def extract(source_dir, age_gender=False, exif=False):

    from sklearn.decomposition import PCA

    global output_dir, network_dir, face_dir, detector
    output_dir=os.path.join(Path(source_dir), "Face Network/")
    network_dir=os.path.join(output_dir, "Data/")
    face_dir=os.path.join(output_dir, "Faces/")

    overwrite(output_dir)
    overwrite(network_dir)
    overwrite(face_dir)


    img_list=makelist(['.jpg', '.jpeg'], source_dir)
    all_images=pd.DataFrame()

    count=len(img_list)
    print("Analyzing {} images".format(count))

    cpus=joblib.cpu_count()-1

    rows=Parallel(n_jobs=cpus)(delayed(crop_face)(a, face_dir, age_gender, exif) for a in img_list)

    all_images=pd.concat(rows)

    if exif:
        all_images['EXIF DateTimeOriginal']=pd.to_datetime(all_images['EXIF DateTimeOriginal'],  format='%Y:%m:%d %H:%M:%S', errors='coerce')


    lowd = PCA(n_components=50).fit_transform(list(all_images['encoding']))
    all_images['encoding']=list(lowd)

    all_images.to_hdf(network_dir+'FaceDatabase.h5', 'index', 'w',complevel=9)    

    print("Face images stored in:", network_dir)

    return all_images


def makelist(extensions, source_dir):
    templist=[]
    for subdir, dirs, files in os.walk(source_dir):
        dirs[:] = [d for d in dirs if d not in 'Faces']
        for file in files:
            for extension in extensions:
                if extension in os.path.join(subdir, file):
                    f=os.path.join(subdir, file)
                    templist.append(f)
    return templist



def crop_face(image_path, face_dir, age_gender=False, exif=False):
    try:

        img_name=image_path.split('/')[-1]

        img = cv2.imread(str(image_path), 1)
        
        if img is not None:
            h, w, _ = img.shape
            r = 1080 / max(w, h)
            img=cv2.resize(img, (int(w * r), int(h * r)), interpolation = cv2.INTER_AREA)

        
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), 224, 224, 3))

        rows=pd.DataFrame()

        if exif:
            keys = ['Image Make', 'Image Model', 'EXIF DateTimeOriginal', 'EXIF BodySerialNumber']

            f = open(image_path, 'rb')
            tags = exifread.process_file(f, details=False)
            exif_row = {key: tags[key] for key in keys if key in tags.keys()}
            if len(exif_row)==0:
                exif_row=dict.fromkeys(keys,np.nan)

        if len(detected) > 0:
            for i, d in enumerate(detected):
                margin=0.4
                face_img_name="face{}_{}".format(str(i+1), img_name)
                
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                
                crop_face=img[yw1:yw2 + 1, xw1:xw2 + 1]
                encoding = face_recognition.face_encodings(crop_face)
                
                faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (224, 224), interpolation = cv2.INTER_AREA)

                if len(encoding)==0:
                    break

                img_row = dict({
                    'img_path':image_path,
                    'img_name':img_name,
                    'face_name':face_img_name,
                    'encoding': encoding[0],
                    'age':0,
                    'gender':0
                    }) 

                if age_gender:
                    # predict ages and genders of the detected faces
                    results = model.predict(faces)
                    predicted_genders = results[0]
                    ages = np.arange(0, 101).reshape(101, 1)
                    predicted_ages = results[1].dot(ages).flatten()

                    age=int(predicted_ages[i])
                    gender=predicted_genders[i][0]


                    age_gender_dict = dict({
                        'age':age,
                        'gender':gender
                        }) 

                    img_row={**img_row,**age_gender_dict}

                if exif:
                    img_row={**img_row,**exif_row}


                rows=rows.append(img_row, ignore_index=True)
                cv2.imwrite(face_dir+face_img_name, crop_face)

            return rows
    except:
        print('image skipped')



def match(row, results, core=False):

    # To assess the quality of the clusters, this function calculates the cosine distance between facial encodings within the same cluster.

    if row['cluster']>=0:

        #get the facial encoding and cluster ID of the reference face
        face=row['encoding']
        cluster=row['cluster']

        # Get the face that is most similar to the other faces in the cluster (the "best" face)
        if core:
            sub=results[results['cluster']==cluster].sort_values(by='cluster_distance',ascending=True).iloc[0]
            sub_encoding=[np.array(sub['encoding'])]

        # Isolate faces in the same cluster as the reference face
        else:
            sub=results[results['cluster']==cluster]
            sub_encoding=list(sub['encoding'])

        # calculate the mean cosine distance between the reference face and all the other faces in this cluster
        # if core=True, calculate the cosine distance between ther reference face and the "best" face in the cluster
        matches = face_recognition.face_distance(face, sub_encoding)
        mean_score=int(np.mean(matches)*100)
    else:
        mean_score=np.NaN

    return mean_score


def match_stragglers(row, df):

    #get the facial encoding and cluster ID of the reference face
    face=row['encoding']

    encodings=list(df['encoding'])

    scores = face_recognition.face_distance(face, encodings)
    matches = pd.DataFrame({'score':scores,'face_name':df['face_name'],'cluster': df['cluster']})

    top=matches.sort_values(by='score',ascending=True).reset_index(drop=True)

    print(int(top.at[0,'score']*100), row['face_name'], top.at[0,'face_name'])

    return top.at[0,'face_name']





def cluster(source_dir, algorithm='DBSCAN', initial_eps=0.44, iterations=1, max_distance=50, mosaic=True, plot=False):


    from sklearn.cluster import DBSCAN
    from sklearn.cluster import OPTICS
    from sklearn.cluster import AgglomerativeClustering
    from sklearn import preprocessing

    global network_dir, face_db, cluster_dir, output_dir

    output_dir=os.path.join(source_dir, "Face Network/")
    network_dir=os.path.join(output_dir, "Data/")
    face_db=pd.read_hdf(network_dir+"FaceDatabase.h5")
    cluster_dir=os.path.join(output_dir, "Clusters/")
    face_dir=os.path.join(output_dir, "Faces/")

    if algorithm=='chinese_whispers':
        print('EPS: ',initial_eps)
        final_results=chinese_whispers(source_dir, threshold=initial_eps, mosaic=mosaic, plot=plot)
        return final_results

    # Create empty df to store results
    final_results=pd.DataFrame()
    
    exit=False


    for i in range(1,iterations+1):

        print('Iteration {}, Algorithm:{}, EPS: {}'.format(i,algorithm,initial_eps))
        
        encodings=list(face_db['encoding'])
        face_names=list(face_db['face_name'])
        img_names=list(face_db['img_name'])


        if algorithm=='OPTICS':
            clt = OPTICS()
            clt.fit(encodings)
            exit=True

        if algorithm=='DBSCAN':

            # Decrease EPS by 0.01 each iteration 
            eps=initial_eps-(i/100)
            clt = DBSCAN(eps=eps, min_samples=3, n_jobs=-1, metric='euclidean', algorithm='kd_tree')
            clt.fit(encodings)

        if algorithm=='AHC':
            eps=3-.2
            clt = AgglomerativeClustering(distance_threshold=eps, compute_full_tree=True, n_clusters=None)

            # Conduct clustering and save results to a dataframe
            model=clt.fit(encodings)
            clt.labels_=clt.labels_#+1

            #plot_dendrogram(model, img_names)


        results=pd.DataFrame({'face_name':face_names, 'img_name':img_names, 'cluster':clt.labels_, 'encoding':encodings})

               
        def parallel_apply(chunk, df, core=False):
            if core:
                chunk['cluster_distance_core']=chunk.apply(lambda x: match(x, df, core=True), axis=1)
            else:
                chunk['cluster_distance']=chunk.apply(lambda x: match(x, df), axis=1)
            return chunk

        cpus=joblib.cpu_count()-1
        df_split = np.array_split(results, cpus)

        rows=Parallel(n_jobs=cpus)(delayed(parallel_apply)(chunk, results) for chunk in df_split)
        results=pd.concat(rows)

        rows=Parallel(n_jobs=cpus)(delayed(parallel_apply)(chunk, results, core=True) for chunk in df_split)
        results=pd.concat(rows)


        # Small clusters and faces with high cosine distance (bad matches) are assigned to a bin cluster with ID -2
        results['cluster']=np.where(results['cluster_distance_core']>max_distance+10,-2,results['cluster'])
        counts=results.groupby('cluster')['face_name'].count().reset_index().rename(columns={'face_name':'count'})
        results=results.merge(counts, how='left',on='cluster')
        results['cluster']=np.where(results['count']<5,-2,results['cluster'])
        results=results.drop(columns='count')

        # Calculate the median cosine distance and percentage of outliers for each cluster. 
        outliers=results.groupby('cluster')[['cluster_distance_core']].agg({'cluster_distance_core':'median'}).reset_index().rename(columns={'cluster_distance_core':'cluster_distance_mean'})
        results=results.merge(outliers, how='left',on='cluster')

        # Assign clusters with a high average cosine distance and those in the bin clusters (-1, -2) to face_db for reanalysis
        
        # Add faces in clusters with low average cosine distance (<40) to final output
        face_db=results[(results['cluster_distance_mean']>max_distance) | (results['cluster']<0)]
        results=results[(results['cluster_distance_mean']<=max_distance) & (results['cluster']>=0)]

        # Count the number of images in each cluster
        counts=results.groupby('cluster')['face_name'].count().reset_index().rename(columns={'face_name':'count'})
        results=results.merge(counts, how='left',on='cluster')
        
        # Generate a cluster code; the first four numbers indicate the number of the iteration, followed by the cluster ID.
        results['cluster']=results['cluster'].apply(lambda x: int((str(i)*4 )+ str(x)))
        final_results=final_results.append(results)

        print("Matched: ", len(final_results),"(+{})".format(len(results)))
        print("Unmatched: ", len(face_db))

        #exit=True
        # When no new matches are found, switch to a more flexible clustering algorithm for the final pass.
        # OPTICS allows for clusters of varying densities. 

        if i>(iterations-1)/2:
            algorithm='DBSCAN'

        #if (len(results) ==0 or i==iterations-1):
        #    algorithm='OPTICS'

        if (len(results) ==0 or len(face_db)==0):
            exit=True

        if exit:
            break

    face_db['cluster']=-2
    final_results=final_results.append(face_db).sort_values(by='count',ascending=False)
    
    le=preprocessing.LabelEncoder()
    le.fit(final_results['cluster'])
    final_results['cluster']=le.transform(final_results['cluster'])

    final_results.reset_index(inplace=False)
    final_results.to_hdf(network_dir+'FaceDatabase.h5', 'index', 'w',complevel=9)    

    if mosaic:
        # build a mosaic of face tiles for each cluster
        overwrite(cluster_dir)
        clusters=final_results['cluster'].unique().tolist()
        clusters = [ elem for elem in clusters if elem > 0]
        if mosaic=='streamlit':
            clusters = [ elem for elem in clusters if elem < 4]

        cpus=joblib.cpu_count()-1
        rows=Parallel(n_jobs=cpus)(delayed(build_mosaic)(cluster,final_results,face_dir,cluster_dir) for cluster in clusters)

    return final_results





def chinese_whispers(source_dir, threshold=0.55, iterations=20, mosaic=True, plot=False):
    """ Chinese Whispers Algorithm
    Modified from Alex Loveless' implementation,
    http://alexloveless.co.uk/data/chinese-whispers-graph-clustering-in-python/
    Inputs:
        encoding_list: a list of facial encodings from face_recognition
        threshold: facial match threshold,default 0.6
        iterations: since chinese whispers is an iterative algorithm, number of times to iterate
    Outputs:
        sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
            sorted by largest cluster to smallest
    """

    output_dir=os.path.join(source_dir, "Face Network/")
    network_dir=os.path.join(output_dir, "Data/")
    face_db=pd.read_hdf(network_dir+"FaceDatabase.h5")
    cluster_dir=os.path.join(output_dir, "Clusters/")
    face_dir=os.path.join(output_dir, "Faces/")

    #face_db=face_db[:1000]

    encodings= list(face_db['encoding'])
    image_paths=list(face_db['face_name'])


    from random import shuffle
    import networkx as nx
    # Create graph
    nodes = []
    edges = []

    if len(encodings) <= 1:
        print ("No enough encodings to cluster!")
        return []

    import time

    start = time.time()

    '''
    for idx, face_encoding_to_check in enumerate(encodings):
        # Adding node of facial encoding

        node_id = idx+1

        print(node_id)

        # Initialize 'cluster' to unique value (cluster of itself)
        node = (node_id, {'cluster': image_paths[idx], 'path': image_paths[idx], 'ID':image_paths[idx].split('_')[1]})
        nodes.append(node)

        # Facial encodings to compare
        if (idx+1) >= len(encodings):
            # Node is last element, don't create edge
            break

        compare_encodings = encodings[idx+1:]
        distances = face_recognition.face_distance(compare_encodings, face_encoding_to_check)
        encoding_edges = []
        for i, distance in enumerate(distances):
            if distance < threshold:
                # Add edge if facial match
                edge_id = idx+i+2
                encoding_edges.append((node_id, edge_id, {'weight': distance}))

        edges = edges + encoding_edges
    '''

    def compare(idx, face_encoding_to_check, edges):
                # Adding node of facial encoding
        node_id = idx+1

        # Initialize 'cluster' to unique value (cluster of itself)
        node = (node_id, {'cluster': image_paths[idx], 'path': image_paths[idx], 'ID':image_paths[idx].split('_')[1]})
        nodes.append(node)

        compare_encodings = encodings[idx+1:]

        #print(node)

        distances = face_recognition.face_distance(compare_encodings, face_encoding_to_check)
        encoding_edges = []
        for i, distance in enumerate(distances):
            if distance < threshold:
                # Add edge if facial match
                edge_id = idx+i+2
                encoding_edges.append((node_id, edge_id, {'weight': distance}))

        edges = edges + encoding_edges

        return edges


    for idx, face_encoding_to_check in enumerate(encodings):
        edges=compare(idx, face_encoding_to_check, edges)


    #print('Check 1', time.time() - start)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Iterate
    for _ in range(0, iterations):
        cluster_nodes = G.nodes()
        #shuffle(cluster_nodes)
        for node in cluster_nodes:
            neighbors = G[node]
            clusters = {}

            for ne in neighbors:
                if isinstance(ne, int):
                    if G.nodes[ne]['cluster'] in clusters:
                        clusters[G.nodes[ne]['cluster']] += G[node][ne]['weight']
                    else:
                        clusters[G.nodes[ne]['cluster']] = G[node][ne]['weight']

            # find the class with the highest edge weight sum
            edge_weight_sum = 0
            max_cluster = 0
            #use the max sum of neighbor weights class as current node's class
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster

            # set the class of target node to the winning local class
            G.nodes[node]['cluster'] = max_cluster

    if plot:
        from itertools import count
        import matplotlib.pyplot as plt

        # get unique groups
        groups = set(nx.get_node_attributes(G,'ID').values())
        mapping = dict(zip(sorted(groups),count()))
        nodes = G.nodes()
        colors = [mapping[G.nodes[n]['ID']] for n in nodes]

        pos = nx.spring_layout(G)

        nx.draw_networkx_edges(G, pos, alpha=0.2)
        nx.draw_networkx_nodes(G, pos, node_size=5, node_color=colors, cmap=plt.cm.jet)
        plt.show()

    clusters = {}

    # Prepare cluster output
    for (_, data) in G.nodes.items():
        cluster = data['cluster']
        path = data['path']

        #print(cluster, path)
        if cluster:
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(path)

    # Sort cluster output
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    length=[]

    cluster_master=pd.DataFrame()
    count=0
    for cluster in sorted_clusters:
        count+=1
        cluster_df=pd.DataFrame({'cluster':count,'face_name':cluster})
        cluster_master=cluster_master.append(cluster_df)

    if 'cluster' in face_db:
        face_db=face_db.drop(columns=['cluster'])

    if 'count' in face_db:
        face_db=face_db.drop(columns=['count'])


    face_db=face_db.merge(cluster_master, on='face_name',how='left')
    face_db['cluster']=face_db['cluster'].fillna(-1).astype(int)
    counts=face_db.groupby('cluster')['face_name'].count().reset_index().rename(columns={'face_name':'count'})
    face_db=face_db.merge(counts, how='left',on='cluster')

    print(face_db['cluster'].unique())
    face_db.to_hdf(network_dir+'FaceDatabase.h5', 'index', 'w',complevel=9)

    if mosaic:
        # build a mosaic of face tiles for each cluster
        overwrite(cluster_dir)
        clusters=face_db['cluster'].unique().tolist()
        clusters = [ elem for elem in clusters if elem > 0]
        cpus=joblib.cpu_count()-1
        rows=Parallel(n_jobs=cpus)(delayed(build_mosaic)(cluster,face_db,face_dir,cluster_dir) for cluster in clusters) 

    return face_db


#def pca_plot(source_dir):


def network(source_dir, scale=10, filter_years=False):
    
    from pyvis.network import Network
    import networkx as nx

    global network_dir, face_db, face_dir, output_dir

    output_dir=os.path.join(source_dir, "Face Network/")
    face_dir=os.path.join(output_dir, "Faces/")
    network_dir=os.path.join(output_dir, "Data/")
    face_db=pd.read_hdf(network_dir+"FaceDatabase.h5")
    
    face_db['ID']=face_db['img_name'].str.split('_').str[0]

    #discard faces that were not matched into clusters
    face_db=face_db[face_db['cluster']>0]

    try:
        face_db['year']=pd.DatetimeIndex(face_db['EXIF DateTimeOriginal']).year.fillna(0)
    except:
        face_db['year']=np.nan
    
    if filter_years:
        face_db=face_db[face_db['year']>0]
    #face_db=face_db[face_db['gender']<.2]

    #group the dataframe by image, creating lists of faces in each image
    images=face_db.groupby('img_name')['cluster'].apply(list).reset_index().rename(columns={'cluster':'connections'})
    face_db=pd.merge(face_db,images,how='left',on='img_name')

    #group the dataframe by cluster, creating lists of co-appearances with other clusters
    clusters=face_db.groupby('cluster')['connections'].apply(tuple).reset_index()
    clusters['connections']=clusters['connections'].apply(lambda x: list([item for sublist in x for item in sublist]))
    info=face_db.groupby('cluster').first().reset_index().drop(columns=['connections'])
    exp=clusters.explode(column='connections')
    exp=exp.merge(info, how='left',on='cluster')

    if "age" and "gender" in face_db.columns:
        age_gender=face_db.groupby('cluster')[['age','gender']].mean().reset_index()
        exp=exp.drop(columns=['age','gender'])
        exp=exp.merge(age_gender, how='left',on='cluster')

    exp['edge_count']=1
    exp['edge']=exp.apply(lambda x: tuple([x['cluster'], x['connections']]),axis=1)
    exp['total_connections']=np.where(exp['cluster']!=exp['connections'], 1,0)

    weight=exp.groupby('edge')['edge_count'].count().reset_index().rename(columns={'edge_count':'weight'})
    size=exp[['cluster','edge_count','total_connections', 'year']].groupby('cluster').agg({'edge_count':'count', 'total_connections':'sum','year':'min'}).reset_index().rename(columns={'edge_count':'size', 'year':'min_year'})


    exp=exp.drop_duplicates(subset=['edge'], keep='first').drop(columns=['total_connections'])
    
    exp=pd.merge(exp,weight,on='edge',how='left')
    exp=pd.merge(exp,size,on='cluster',how='left').sort_values(by='total_connections',ascending=False)

    from sklearn import preprocessing

    edge_color='EXIF BodySerialNumber'

    exp[edge_color]=exp[edge_color].astype(str)
    le=preprocessing.LabelEncoder()
    le.fit(exp[edge_color])

    exp[edge_color]=le.transform(exp[edge_color])

    import matplotlib.pyplot as plt

    #plt.hist(exp['year'])
    #plt.show()

    g = nx.Graph()

    net = Network(height='800px', width='800px', bgcolor='#0d1217')
    #net.show_buttons()

    #net.barnes_hut(spring_length=200)
    #net.enable_physics(False)

    exp=exp.sort_values(by='total_connections',ascending=False)

    for index, row in exp.iterrows():
        src=str(row['cluster'])
        s=np.log10(row['size'])*scale
        connections=str(row['total_connections'])+'<br>'
        image_count=str(int(row['count']))+'<br>'
        path=face_dir+str(row['face_name'])

        try:
            gender=round(row['gender'], 3)
            age=int(row['age'])
            if gender>0.2:
                color='#FFA500'
                gender=1

            else:
                gender=0
                color='#3437eb'

        except:
            gender='N/A'
            age='N/A'
        color='#FFA500'

        year=int(row['min_year'])



        tag=("Individual ID: "+src+'<br> Connections: '+connections+'Images: '+image_count+'Gender: '+ str(gender))

        net.add_node(src, label=src,size=s, title=tag, shape='circularImage',image=path, borderWidth=5, color=color)
        g.add_node(src, label=src, gender=gender, age=age, size=s, year=year)

    for index, row in exp.iterrows():
        src = str(row['cluster'])
        dst = str(row['connections'])
        w = row['weight']
        color=int(row[edge_color])

        if src !=dst:
            net.add_edge(src, dst, value=w, title=w)
            g.add_edge(src, dst, weight=w, color=color)

    # filter junk
    #junk = ['13', '234', '2767', '1'] #when eps=.39
    #junk = ['21', '14', '1683', '124', '2812', '2839', '1'] #when eps=.4
    #g.remove_nodes_from(junk)

    edgelist=pd.DataFrame(g.edges, columns=['from','to'])
    edgelist.to_csv(network_dir+'Edges.csv', index=False)
    net.save_graph(network_dir+'Image_Network.html')
    nx.write_gpickle(g, network_dir+'Image_Network.gpickle')

    print("Network graph created in: "+network_dir+'Image_Network.html')
    
    return g

def plot_network(photo_dir, gtype='PICAN'):
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.pyplot import figure

    mpl.rcParams.update({'font.size': 14})

    network_path=photo_dir+'/Face Network/Data/Image_Network.gpickle'
    g=nx.read_gpickle(network_path)

    if gtype=='preferential':
        g=network(photo_dir, filter_years=True)

    junk = ['13', '234', '2767', '1'] #when eps=.39
    
    g.remove_nodes_from(junk)

    figure(figsize=(9, 7), dpi=80)


    if gtype=='PICAN':
        G0=g
        print(G0)
        scale=5


    if gtype in ['gender','leaders', 'preferential', 'communities', 'small_world']:

        scale=5

        if gtype=='preferential':
            scale=15

        if gtype=='small_world':
            random = nx.fast_gnp_random_graph(450, 0.0056, seed=1)
            Gcc = sorted(nx.connected_components(random), key=len, reverse=True)
            random_lcc = random.subgraph(Gcc[0])

        Gcc = sorted(nx.connected_components(g), key=len, reverse=True)
        G0 = g.subgraph(Gcc[0])
        
        #G0=g
        #G0.remove_nodes_from(list(nx.isolates(G0)))

        pos = nx.kamada_kawai_layout(G0, scale=100)


    if gtype=='leaders':
        scale=15


    edges = G0.edges()
    nodes = G0.nodes()

    colors = [G0[u][v]['color'] for u,v in edges]
    weights = [G0[u][v]['weight'] for u,v in edges]

    size = [v * scale for v in nx.get_node_attributes(G0,'size').values()]

    _, degree= zip(*G0.degree(weight='weight'))
    degree=[np.log10(1+v)  for v in degree]
    gender=list(nx.get_node_attributes(G0,'gender').values())
    year=list(nx.get_node_attributes(G0,'year').values())
    labels=nx.get_node_attributes(G0,'label')
    shapes = ["o"]*len(g.nodes)

    if gtype=='gender':

        cmap=plt.cm.bwr

        options = {'node_size':size, 'node_color':gender, 'cmap':cmap, 'edgecolors':'grey'}

        plt.title('Distribution of Male and Female PKK Members')

    if gtype=='preferential':

        pos = nx.spring_layout(G0, k=0.05, weight='weight')

        cmap=plt.cm.magma

        options = {'node_size':size, 'node_color':year, 'cmap':cmap, 'edgecolors':'grey', 'vmin':2002, 'vmax':2018}

        betweenness=pd.DataFrame(nx.betweenness_centrality(G0).items()).rename(columns={0 : 'cluster',1 : 'betweenness'})

        b_centrality=list(betweenness['betweenness'])

        betweenness=betweenness.sort_values(by='betweenness', ascending=False).reset_index(drop=True)

        print(betweenness[:8])

        betweenness.index += 1 

        labels=betweenness[:5][['cluster']].to_dict()['cluster']
        labels = {y:x for x,y in labels.items()}

        nx.draw_networkx_labels(G0, pos, labels, font_weight='bold', font_color='white',font_size=14) 

        
        cbar_range=(range(2002,2020))
        cbar_title='Year Photograph was Taken'
        title='Preferential attachment over time using EXIF data'


    if gtype=='PICAN':

        cmap=plt.cm.turbo

        pos = nx.spring_layout(G0, k=0.07, weight='weight')
        options = {'node_size':size, 'node_color':degree, 'cmap':cmap, 'edgecolors':'grey'}

        cbar_range=range(0,60)
        cbar_title='Node Degree'
        title='PKK Image Co-Appearence Network (PICAN)'


    if gtype=='communities':
        
        cmap=plt.cm.tab10

        communities=list(nx.algorithms.community.greedy_modularity_communities(G0, weight='weight'))

        count=0
        community_df=pd.DataFrame()
        for i in communities:
            count+=1
            group=pd.DataFrame(zip([count]*len(i),i))
            community_df=community_df.append(group)
        
        community_df=community_df.rename(columns={0:'community',1:'node'}).sort_values(by='node')

        options = {'node_size':size, 'node_color':community_df['community'], 'cmap':cmap, 'edgecolors':'grey', 'vmin':1, 'vmax':20} 

        cbar_range=range(1,20)
        cbar_title='Betweenness Centrality'
        title='Betweenness Centrality of PKK Leadership'

    if gtype=='leaders':
        
        cmap=plt.cm.rainbow

        betweenness=pd.DataFrame(nx.betweenness_centrality(G0).items()).rename(columns={0 : 'cluster',1 : 'betweenness'})

        b_centrality=list(betweenness['betweenness'])

        betweenness=betweenness.sort_values(by='betweenness', ascending=False).reset_index(drop=True)

        betweenness.index += 1 

        options = {'node_size':size, 'node_color':b_centrality, 'cmap':cmap, 'edgecolors':'grey', 'vmin':0, 'vmax':0.15} 

        labels=betweenness[:15][['cluster']].to_dict()['cluster']
        labels = {y:x for x,y in labels.items()}

        print(betweenness, G0.nodes)
        cbar_range=range(0,15)
        cbar_title='Betweenness Centrality'
        title='Betweenness Centrality of PKK Leadership'

        nx.draw_networkx_labels(G0, pos, labels, font_weight='bold', font_size=14) 

    if gtype=='small_world':

        cmap=plt.cm.jet

        _, r_degree= zip(*random_lcc.degree(weight='weight'))
        r_degree=[np.log10(1+v)  for v in r_degree]

        pos_0 = nx.spring_layout(G0)
        pos_random = nx.spring_layout(random_lcc)
        
        pos_0_circular = nx.circular_layout(G0)
        pos_random_circular = nx.circular_layout(random_lcc)    
           
        options = {'node_size':10, 'node_color':degree, 'cmap':cmap, 'edgecolors':'grey'}
        
        r_options = {'node_size':10, 'node_color':r_degree, 'cmap':cmap, 'edgecolors':'grey'}

        fig, axes = plt.subplots(ncols=2, nrows=2)
        ax = axes.flatten()

        edge_vis = nx.draw_networkx_edges(G0, pos_0, width=weights, alpha=.2, ax=ax[0])
        node_vis=nx.draw_networkx_nodes(G0, pos_0, node_shape='o', **options, ax=ax[0]) 

        nx.draw_networkx_edges(random_lcc, pos_random, alpha=.2, ax=ax[1])
        nx.draw_networkx_nodes(random_lcc, pos_random, node_shape='o',**r_options, ax=ax[1]) 

        edge_vis = nx.draw_networkx_edges(G0, pos_0_circular, width=weights, alpha=.2, ax=ax[2])
        node_vis=nx.draw_networkx_nodes(G0, pos_0_circular, node_shape='o', **options, ax=ax[2]) 

        nx.draw_networkx_edges(random_lcc, pos_random_circular, alpha=.2, ax=ax[3])
        nx.draw_networkx_nodes(random_lcc, pos_random_circular, node_shape='o',**r_options, ax=ax[3]) 

        ax[1].set_title('Erdős–Rényi Random Graph')
        ax[0].set_title('PICAN')

    if gtype in ['PICAN','leaders','preferential']:
        pc = mpl.collections.PathCollection(node_vis, cmap=cmap)
        pc.set_array(cbar_range)
        cbar=plt.colorbar(pc)
        cbar.set_label(cbar_title)

    if gtype !='small_world':
        edge_vis = nx.draw_networkx_edges(G0, pos, width=weights, alpha=.5)
        node_vis=nx.draw_networkx_nodes(G0, pos, node_shape='o', **options) 
        plt.title(title)
    
    plt.tight_layout()
    plt.show()



def centrality_plot(df):

    import matplotlib.pyplot as plt

    df=df[:15]

    print(df)

    labels = list(df['cluster'])
    labels=['Erdal','Karayılan','Kalkan','Gurhan','Welat','Akdoğan','Celik','Bayık','Kaya','Unknown','Çewlik','Goyi','Güney','Taylan','Kaytan']

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width/2, df['eigenvector'], width, label='Eigenvector Centrality')
    rects2 = ax.bar(x + width/2, df['betweenness'], width, label='Betweenness Centrality')
    rects3 = ax.bar(x + (width/2)*3, df['degree'], width, label='Degree Centrality')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Centrality')
    ax.set_title('15 most central nodes')
    print(labels)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()
    fig.tight_layout()

    plt.show()


def network_analysis(photo_dir, plot=None, graph=False, cliques=False, communities=False):
    
    import networkx as nx
    from functools import reduce
    import matplotlib.pyplot as plt
    import math

    network_path=str(photo_dir)+'/Face Network/Data/Image_Network.gpickle'

    face_db=pd.read_hdf(str(photo_dir)+"/Face Network/Data/FaceDatabase.h5")
    face_db=face_db[face_db['cluster']>1]

    #print(face_db[face_db['cluster']==2]['EXIF DateTimeOriginal'])

    if graph:
        G=graph

    else:
        G=nx.read_gpickle(network_path)


    junk = ['13', '234', '2767', '1'] #when eps=.39

    G.remove_nodes_from(junk)

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(components[0])

    #G.remove_nodes_from(list(nx.isolates(G)))
    
    degree_sequence = sorted([d for n, d in G0.degree()], reverse=True) # used for degree distribution and powerlaw test

    import powerlaw
    fit = powerlaw.Fit(degree_sequence, discrete=True)

    '''
    print(fit.distribution_compare('power_law', 'stretched_exponential'))
    print(fit.distribution_compare('power_law', 'lognormal'))
    print(fit.distribution_compare('power_law', 'exponential'))
    print(fit.power_law.alpha, fit.power_law.sigma, fit.power_law.D)
    '''


    if plot=='powerlaw':
        fig4 = fit.plot_ccdf(linewidth=2,
            label='Empirical Data')
        fit.power_law.plot_ccdf(ax=fig4, 
            color='red', 
            linestyle='--', 
            label='Power Law (α={})'.format(round(fit.power_law.alpha, 2)))
        fit.lognormal.plot_ccdf(ax=fig4, 
            color='purple', 
            linestyle='--',  
            label='Log Normal')
        '''
        fit.stretched_exponential.plot_ccdf(ax=fig4, 
            color='orange', 
            linestyle='--',  
            label='Stretched Exponential') 
        '''
        plt.xlabel('Node Degree (k)')
        plt.ylabel('P(X≥k)')
        plt.legend()
        plt.show()


    avg_clustering=nx.average_clustering(G0, weight='weight')
    degree_assortativity=nx.degree_assortativity_coefficient(G0, weight='weight')
    attribute_assortativity=nx.attribute_assortativity_coefficient(G0, "gender")
    density=nx.classes.function.density(G0)

    #small world
    shortest_path=nx.average_shortest_path_length(G0)

    #sigma=nx.sigma(G0)
    sigma=2.69
    #degree=pd.DataFrame(nx.degree(G0)).rename(columns={1:'degree'})
    degree=pd.DataFrame(nx.degree_centrality(G0).items()).rename(columns={1:'degree'})
    eigenvector=pd.DataFrame(nx.eigenvector_centrality(G0).items()).rename(columns={1:'eigenvector'})
    betweenness=pd.DataFrame(nx.betweenness_centrality(G0).items()).rename(columns={1:'betweenness'})


    df = reduce(lambda left,right: pd.merge(left,right,on=0), [degree,eigenvector, betweenness])
    df=df.rename(columns={0:'cluster'}).sort_values(by='betweenness', ascending=False)
    df['cluster']=df['cluster'].astype(int)
    if plot=='centrality':
        

        df.to_csv(photo_dir+'/Face Network/Data/Metrics.csv', index=None)

        centrality_plot(df)



    node, degree= zip(*G0.degree(weight='weight'))

    '''
    plt.plot(degree,'.')
    #plt.plot(df['eigenvector'])
    #plt.plot(df['betweenness'])

    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    '''
    if cliques:
        cliques=list(nx.find_cliques(G))

        images=face_db.groupby('img_name')['cluster'].apply(set).reset_index()#.rename(columns={'cluster':'connections'})

        clique_df=pd.DataFrame()
        for clique in cliques:
            if len(clique)>2:
                clique= [int(i) for i in clique]
                pic=images[images['cluster']==set(clique)]['img_name'].unique()
                deg=df[df['cluster'].isin(clique)]['degree']
                
                row={'clique':clique,
                'img_name':pic,
                'stddev':np.std(deg), 
                'mean':deg.mean(),
                'range':deg.max()-deg.min()
                }
                clique_df=clique_df.append(row, ignore_index=True)
        print(clique_df.sort_values(by='stddev')[['stddev','img_name']].dropna())

        clusters=face_db.groupby('cluster')[['gender','count']].mean()
        
        
        men=clusters[clusters['gender']<0.2]
        women=clusters[clusters['gender']>0.2]
        
        import scipy
        from scipy import stats
        ttest=scipy.stats.ttest_ind(women['count'], men['count'], alternative='greater')   
        print(men['count'].mean(), women['count'].mean(), ttest)


    if communities:
        from networkx.algorithms.community.centrality import girvan_newman
        import networkx as nx
        from itertools import chain, combinations
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram

        # get simulated Graph() and Girvan-Newman communities list
        #G = nx.path_graph(10)
        communities = list(nx.community.girvan_newman(G0))

        # building initial dict of node_id to each possible subset:
        node_id = 0
        init_node2community_dict = {node_id: communities[0][0].union(communities[0][1])}
        for comm in communities:
            for subset in list(comm):
                if subset not in init_node2community_dict.values():
                    node_id += 1
                    init_node2community_dict[node_id] = subset

        # turning this dictionary to the desired format in @mdml's answer
        node_id_to_children = {e: [] for e in init_node2community_dict.keys()}
        for node_id1, node_id2 in combinations(init_node2community_dict.keys(), 2):
            for node_id_parent, group in init_node2community_dict.items():
                if len(init_node2community_dict[node_id1].intersection(init_node2community_dict[node_id2])) == 0 and group == init_node2community_dict[node_id1].union(init_node2community_dict[node_id2]):
                    node_id_to_children[node_id_parent].append(node_id1)
                    node_id_to_children[node_id_parent].append(node_id2)

        # also recording node_labels dict for the correct label for dendrogram leaves
        node_labels = dict()
        for node_id, group in init_node2community_dict.items():
            if len(group) == 1:
                node_labels[node_id] = list(group)[0]
            else:
                node_labels[node_id] = ''

        # also needing a subset to rank dict to later know within all k-length merges which came first
        subset_rank_dict = dict()
        rank = 0
        for e in communities[::-1]:
            for p in list(e):
                if tuple(p) not in subset_rank_dict:
                    subset_rank_dict[tuple(sorted(p))] = rank
                    rank += 1
        subset_rank_dict[tuple(sorted(chain.from_iterable(communities[-1])))] = rank

        # my function to get a merge height so that it is unique (probably not that efficient)
        def get_merge_height(sub):
            sub_tuple = tuple(sorted([node_labels[i] for i in sub]))
            n = len(sub_tuple)
            other_same_len_merges = {k: v for k, v in subset_rank_dict.items() if len(k) == n}
            min_rank, max_rank = min(other_same_len_merges.values()), max(other_same_len_merges.values())
            range = (max_rank-min_rank) if max_rank > min_rank else 1
            return float(len(sub)) + 0.8 * (subset_rank_dict[sub_tuple] - min_rank) / range

        # finally using @mdml's magic, slightly modified:
        G0           = nx.DiGraph(node_id_to_children)
        nodes       = G0.nodes()
        leaves      = set( n for n in nodes if G0.out_degree(n) == 0 )
        inner_nodes = [ n for n in nodes if G0.out_degree(n) > 0 ]

        # Compute the size of each subtree
        subtree = dict( (n, [n]) for n in leaves )
        for u in inner_nodes:
            children = set()
            node_list = list(node_id_to_children[u])
            while len(node_list) > 0:
                v = node_list.pop(0)
                children.add( v )
                node_list += node_id_to_children[v]
            subtree[u] = sorted(children & leaves)

        inner_nodes.sort(key=lambda n: len(subtree[n])) # <-- order inner nodes ascending by subtree size, root is last

        # Construct the linkage matrix
        leaves = sorted(leaves)
        index  = dict( (tuple([n]), i) for i, n in enumerate(leaves) )
        Z = []
        k = len(leaves)
        for i, n in enumerate(inner_nodes):
            children = node_id_to_children[n]
            x = children[0]
            for y in children[1:]:
                z = tuple(sorted(subtree[x] + subtree[y]))
                i, j = index[tuple(sorted(subtree[x]))], index[tuple(sorted(subtree[y]))]
                Z.append([i, j, get_merge_height(subtree[n]), len(z)]) # <-- float is required by the dendrogram function
                index[z] = k
                subtree[z] = list(z)
                x = z
                k += 1

        # dendrogram
        plt.figure()
        dendrogram(Z, labels=[node_labels[node_id] for node_id in leaves])
        plt.show()
        #plt.savefig('dendrogram.png')
        quit()

    metrics={'Size':len(G.nodes), 
        'Edges':len(G0.edges),
        'Max Degree': max(degree), 
        'Largest Connected Component Size': len(G0.nodes), 
        'Average Clustering Coefficient': avg_clustering,
        'Density': density,
        'Average Shortest Path': shortest_path,
        'Small-world sigma': sigma,
        'Global assortativity coefficient by degree': degree_assortativity, 
        'Gender assortativity': attribute_assortativity,
        'Degree power-law fit exponent': round(fit.power_law.alpha, 2)}

    #print("\n".join("{!r}: {!r},".format(k, v) for k, v in metrics.items()) )

    return metrics



def build_mosaic(cluster, df, face_dir, cluster_dir):
    from imutils import build_montages

    image_list=df[df['cluster']==cluster]['face_name']#.sort_values(by='cluster_distance_core')
    faces=[]
    for i in image_list:
        image = cv2.imread(face_dir+i)
        faces.append(image)
    dim=int(np.sqrt(len(image_list)))+1
    print(dim)
    mosaic = build_montages(faces, (500, 500), (dim, dim))[0]
    cv2.imwrite(cluster_dir+str(cluster)+'.jpg', mosaic)


def plot_dendrogram(model, img_names):

    from scipy.cluster.hierarchy import dendrogram
    from matplotlib.pyplot import cm
    from scipy.cluster import hierarchy
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    tree=list(zip(model.children_, model.distances_))

    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts])#.astype(float)

    dflt_col = "#808080"

    IDs=[]
    for i in img_names:
        ID='_'.join(i.split('_')[0:-1])
        IDs.append(ID)
    
    ID_set=set(IDs)
    cmap = cm.rainbow(np.linspace(0, 1, len(ID_set)))
    hex_colors=[mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap]

    D_leaf_colors=dict(zip(ID_set, hex_colors))

    link_cols = {}
    for i, i12 in enumerate(linkage_matrix[:,:2].astype(int)):

        c1, c2 = (link_cols[x] if x > len(linkage_matrix) else D_leaf_colors[IDs[x]] for x in i12)
        link_cols[i+1+len(linkage_matrix)] = c1 if c1 == c2 else dflt_col

    dendrogram(linkage_matrix, link_color_func=lambda x: link_cols[x], distance_sort=True, leaf_font_size=10)#, labels=IDs)

    plt.title('Hierarchical Clustering Dendrogram')
    #plt.axhline(eps, color='r')
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.tight_layout()
    plt.show()


def accuracy_assessment(photo_dir, min_count=2, confusion_matrix=False):
    from random import random
    from sklearn import metrics
    import matplotlib.pyplot as plt
    import seaborn as sn


    df=pd.read_hdf(photo_dir+'/Face Network/Data/FaceDatabase.h5')
    df['ID']=df['img_name'].str.split('_').str[0]
    
    #df=df[df['ID']!='cache']
    df['ID']=df.apply(lambda x: unhash(x), axis=1)

    #filter images with more than one face
    df['face_count']=df['face_name'].str.split('_').str[0].str[-1].astype(int)
    multi_face=df.groupby('img_name')['face_count'].max().reset_index()
    df=df.drop(columns=['face_count'])
    df=df.merge(multi_face,how='left',on='img_name')
    df=df[df['face_count']==1]

    df['ID_count']=df['ID']
    clusters=df.groupby('ID').agg({'cluster':lambda x: x.value_counts().index[0], 'ID_count':'count'}).reset_index().rename(columns={'cluster':'cluster_mode'})
    df=df.drop(columns=['ID_count'])
    df=pd.merge(df,clusters, how='left',on='ID')
    df=df[df['count']>min_count]

    clusters=df.groupby('cluster').agg({'ID':lambda x: x.value_counts().index[0]}).reset_index().rename(columns={'ID':'ID_mode'})
    df=pd.merge(df,clusters, how='left',on='cluster')

    from sklearn.metrics.cluster import adjusted_mutual_info_score
    from sklearn.metrics import f1_score

    matched=(len(df[df['cluster']>0])/len(df))

    df=df[df['cluster']>0]

    nmi=adjusted_mutual_info_score(df['ID'], df['cluster'])
    rand=metrics.adjusted_rand_score(df['ID'], df['cluster'])
    homogeneity=metrics.homogeneity_score(df['ID'], df['cluster'])

    if confusion_matrix:
        contingency_matrix = metrics.cluster.contingency_matrix(df['cluster'], df['cluster_mode'])
        fig = plt.figure()
        #fig.set_aspect(1)
        plt.clf()
        res = sn.heatmap(contingency_matrix, vmax=20, cmap='Blues')
        plt.xlabel("Label")
        plt.ylabel("Predicted Label")
        plt.show()


    print("% Matched: {}\nNMI: {}\nRand: {}\nHomogeneity:{}\n".format(matched, nmi, rand, homogeneity))
    return rand, nmi, homogeneity, matched


def unhash(row):
    if row['ID']=='cache':
        i=0
        target=row['img_name'].split('_')[1]
        while True:
            i+=1
            string=("Image"+str(i)).encode('utf-8')
            to_check=hashlib.md5(string)
            hexval=to_check.hexdigest()
            #print(hexval)
            if hexval==target:
                #print(i)
                row['ID']=i
                break
            if i>8000:
                row['ID']='missing'
                break
            
    return int(row['ID'])


def plot_accuracy(photo_dir, bounds=[0,100], xlabel='', **kwargs):
    import matplotlib.pyplot as plt

    accuracy=pd.DataFrame(columns={'max_distance','Matched Images', 'Adjusted Mutual Information','Homogeneity'})

    for i in range(bounds[0],bounds[1]):

        if kwargs['algorithm']=='DBSCAN' or 'chinese_whispers':
            eps=i/100
        if kwargs['algorithm']=='AHC':
            eps=i/10

        clustered=cluster(photo_dir, initial_eps=eps, **kwargs)

        metrics=accuracy_assessment(photo_dir)

        row=dict({'max_distance':i, 'Rand Index':metrics[0], 'Adjusted Mutual Information':metrics[1], 'Percent Matched':metrics[3]})
        accuracy=accuracy.append(row, ignore_index=True)

        plt.clf()
        plt.plot(accuracy['max_distance'], accuracy['Adjusted Mutual Information'], label='Adjusted Mutual Information')
        plt.plot(accuracy['max_distance'],accuracy['Percent Matched'], label='Percent Matched')
        plt.plot(accuracy['max_distance'],accuracy['Rand Index'], label='Rand Index')
        plt.xlabel(xlabel)
        plt.legend()
    plt.show()
    return plt


def removal(iterations=1000, removal_fraction=0.5, strategy='random', hit_list=[]):
    import networkx as nx
    import random

    if strategy=='opportunistic':
        k=removal_fraction
        label=str(removal_fraction)
        hit_list=OG.nodes
    else:
        k=int(OG_nodes*removal_fraction)
        label=str(int(removal_fraction*100))+'%'

    results=[]
    

    for i in range(0,iterations):

        G=nx.read_gpickle(network_path)

        random_removal=random.sample(hit_list, k)

        G.remove_nodes_from(random_removal)
        
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        
        G0 = G.subgraph(components[0])
        
        LCC_size=len(G0.nodes)/OG_LCC

        results.append(LCC_size)


    return pd.DataFrame(results).rename(columns={0:label})



def robustness(photo_dir, strategy='random', hit_list=[]):
    import networkx as nx
    import matplotlib.pyplot as plt

    global network_path, OG_LCC, OG_nodes, OG

    network_path=photo_dir+'/Face Network/Data/Image_Network.gpickle'

    OG=nx.read_gpickle(network_path)
    components = sorted(nx.connected_components(OG), key=len, reverse=True)
    G0 = OG.subgraph(components[0])
    OG_LCC=len(G0.nodes)
    OG_nodes=len(OG.nodes)

    if strategy=='individual':

        G=nx.read_gpickle(network_path)

        G.remove_nodes_from(hit_list)
        
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        
        G0 = G.subgraph(components[0])
        
        LCC_size=len(G0.nodes)/OG_LCC

        decrease=round((1-LCC_size)*100, 2)

        print("The removal of node(s) {} decreases the size of the LCC by {}%".format(hit_list, decrease))

    if strategy=='random':
        random_df=pd.DataFrame()
        for i in range(1,8):
            random=removal(iterations=1000,removal_fraction=i/10, hit_list=OG.nodes)
            random_df=pd.concat([random_df,random], axis=1)
        
        random_df.plot(kind = "kde")
        plt.title('Robustness to Random Node Removal')
        plt.xlabel('Relative LCC size')
        plt.show()

    if strategy=='targeted':
        degree=pd.DataFrame(nx.degree_centrality(G0).items()).rename(columns={1:'degree', 0:'cluster'})  
        leaders=degree.sort_values(by='degree',ascending=False)

        results=[]

        targeted_df=pd.DataFrame()

        for i in range(0,50):

            G=nx.read_gpickle(network_path)
            
            G.remove_nodes_from(leaders['cluster'][:i])
            
            components = sorted(nx.connected_components(G), key=len, reverse=True)
            
            G0 = G.subgraph(components[0])
            
            LCC_size=len(G0.nodes)/OG_LCC

            results.append(LCC_size)

            df=pd.DataFrame(results).rename(columns={0:str(i)})
            
            targeted_df=pd.concat([targeted_df,df], axis=1)

    if strategy=='opportunistic':
        degree=pd.DataFrame(nx.degree_centrality(G0).items()).rename(columns={1:'degree', 0:'cluster'})  
        leaders=list(degree.sort_values(by='degree',ascending=False)['cluster'][:30])

        results=[]

        random_df=pd.DataFrame()

        for i in range(1,10):

            random=removal(iterations=1000,removal_fraction=i*3, hit_list=leaders, strategy='opportunistic')
            random_df=pd.concat([random_df,random], axis=1)
        
        random_df.plot(kind = "kde")
        plt.title('Robustness to Opportunistic Central Node Removal')
        plt.xlabel('Relative LCC size')
        plt.show()



    #network_analysis(photo_dir, graph=G)


def find(source_dir, needle):
    global network_dir, face_db, cluster_dir, output_dir
    output_dir=os.path.join(source_dir, "Face Network/")
    network_dir=os.path.join(output_dir, "Data/")
    face_db=pd.read_hdf(network_dir+"FaceDatabase.h5")
    cluster_dir=os.path.join(output_dir, "Clusters/")
    face_dir=os.path.join(output_dir, "Faces/")
    target_dir=os.path.join(output_dir, "Targets/")

    target_face= np.nan
    matches=[]

    try:
        target_data=crop_face(needle, target_dir)
        target_face=target_data['encoding'][0]
        faces=list(face_db['encoding'])
        face_db['match'] = face_recognition.face_distance(target_face, faces)
        matches=list(face_db[face_db['match']<0.4]['cluster'])
        cluster=max(set(matches), key=matches.count)
    except:
        cluster=0

    return cluster, target_face, len(matches)


def wanted_metrics(source_dir, var):

    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection
    from matplotlib.pyplot import figure

    global network_dir, face_db, cluster_dir, output_dir
    output_dir=os.path.join(source_dir, "Face Network/")
    network_dir=os.path.join(output_dir, "Data/")
    face_db=pd.read_hdf(network_dir+"FaceDatabase.h5")
    cluster_dir=os.path.join(output_dir, "Clusters/")
    face_dir=os.path.join(output_dir, "Faces/")
    target_dir=os.path.join(output_dir, "Targets/")

    wanted_df=pd.read_csv(source_dir+'/Wanted/Wanted_combined.csv')[['cluster_manual','color', 'name','reward', 'birth']].dropna(subset=['cluster_manual'])
    metrics=pd.read_csv(network_dir+'metrics.csv')
    df=wanted_df.merge(face_db, left_on='cluster_manual',right_on='cluster', how='left')
    df=df.merge(metrics, on='cluster', how='left')
    df=df[df['cluster_manual']>0]
    df=df.groupby('cluster_manual').first().fillna(0)


    def destring(x):
        l=[int(s) for s in x.split() if s.isdigit()]
        try:
            l=2021-l[0]
        except:
            l=2021-[int(s) for s in x.split()[0].split('-') if s.isdigit()][0]
        return l   

    df['birth_year']=df['birth'].apply(lambda x: destring(x))


    

    colors= {'red':'red','blue':'blue','green':'green','orange':'orange','grey':'grey'}
    mapping={10000:'#d42a2d',3000:'blue',2000:'green',1000:'#fe802b',500:'grey'}

    if var=='age':
        df['var']=df['birth_year']
    else:
        df['var']=(df['degree']/df['count'])+(np.random.uniform(low=0, high=0.00015, size=(len(df),)))


    #print(df[['name','color','var']].sort_values(by='var'))

    df.to_csv(source_dir+'/Wanted/Wanted_metrics.csv')
    f = plt.figure()
    f.set_figwidth(7)
    f.set_figheight(5)

    ax = sns.boxplot(x="reward", y="var", data=df, palette=mapping, saturation=0.5, fliersize=0, order=mapping)
    sns.stripplot(x="reward", y="var", size=5, data=df, palette=mapping, order=mapping, linewidth=1.5, alpha=0.75)    
    plt.xlabel("Reward (₺, 000)")
    plt.ylabel("Standardized Degree Centrality")
    plt.title('Network Centrality and Wanted Level')

    plt.show()



def serial_mosaic(source_dir, serial):
    from imutils import build_montages

    global network_dir, face_db, cluster_dir, output_dir
    output_dir=os.path.join(source_dir, "Face Network/")
    network_dir=os.path.join(output_dir, "Data/")
    face_db=pd.read_hdf(network_dir+"FaceDatabase.h5")
    cluster_dir=os.path.join(output_dir, "Clusters/")
    target_dir=os.path.join(output_dir, "Targets/")

    face_db['face_count'] = face_db['img_name'].groupby(face_db['img_name']).transform('count')
    face_db=face_db[face_db['face_count']>1]
    face_db=face_db.groupby('img_name').first().reset_index().sort_values(by='cluster', ascending=False)
    face_db['EXIF BodySerialNumber']=face_db['EXIF BodySerialNumber'].astype(str)
    face_db['Image Model']=face_db['Image Model'].astype(str)

    face_db.groupby('img_name')
    #image_list=face_db[face_db['EXIF BodySerialNumber']==serial]['img_name']
    image_list=face_db[face_db['Image Model']==serial]['img_name']
    faces=[]
    for i in image_list:
        image = cv2.imread(source_dir+'/'+i)
        faces.append(image)
    dim=int(np.sqrt(len(image_list)))+1
    print(dim)
    mosaic = build_montages(faces, (700, 500), (dim, dim))[0]
    cv2.imshow('mosaic', mosaic)
    #cv2.imwrite(cluster_dir+str(serial)+'.jpg', mosaic)


def camera_subnet(photo_dir):
    
    import networkx as nx
    from functools import reduce
    import matplotlib.pyplot as plt
    import math

    network_path=photo_dir+'/Face Network/Data/Image_Network.gpickle'
    g=nx.read_gpickle(network_path)

    face_db=pd.read_hdf(photo_dir+"/Face Network/Data/FaceDatabase.h5")
    face_db=face_db[face_db['cluster']>1]
    face_db['Image Model']=face_db['Image Model'].astype(str).dropna()
    cameras=face_db.groupby('Image Model').count().sort_values(by='count', ascending=False).reset_index()['Image Model']
    for camera in cameras[1:]:
        sub=face_db[face_db['Image Model']==camera]
        subg=g.subgraph(sub['cluster'].astype(str))

        print(camera, len(sub))
        nx.draw(subg, node_size=15) 
        serial_mosaic(photo_dir, camera)
        plt.show()

    #print(cameras)


def pathlength(photo_dir):
    import networkx as nx
    network_path=photo_dir+'/Face Network/Data/Image_Network.gpickle'
    g=nx.read_gpickle(network_path)

    print(nx.shortest_path(g, source='2', target='2'))
    print(nx.shortest_path(g, source='2', target='3'))
    print(nx.shortest_path(g, source='2', target='4'))
