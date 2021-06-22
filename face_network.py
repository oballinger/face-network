import os
import cv2
import json
import dlib
import shutil
import joblib
import exifread
import numpy as np
import pandas as pd
import face_recognition
from pathlib import Path
from omegaconf import OmegaConf
from joblib import Parallel, delayed
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import get_file
from tensorflow.keras.optimizers import SGD, Adam
import warnings

def get_model(cfg):
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = '6d7f7b7ced093a8b3ef6399163da6ece'

margin = 0.4

weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="pretrained_models",
                       file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

# load model and weights
model_name, img_size = Path(weight_file).stem.split("_")[:2]
img_size = int(img_size)
cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
model = get_model(cfg)
model.load_weights(weight_file)
detector = dlib.get_frontal_face_detector()

def overwrite(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)


def extract(source_dir, age_gender=False, exif=False):

    global output_dir, network_dir, face_dir, detector
    output_dir=os.path.join(Path(source_dir), "Face Network/")
    network_dir=os.path.join(output_dir, "Data/")
    face_dir=os.path.join(output_dir, "Faces/")

    overwrite(output_dir)
    overwrite(network_dir)
    overwrite(face_dir)

    img_list=makelist('.jpg', source_dir=source_dir)
    all_images=pd.DataFrame()

    count=len(img_list)
    print("Analyzing {} images".format(count))

    cpus=joblib.cpu_count()-1
    rows=Parallel(n_jobs=cpus)(delayed(crop_face)(a,face_dir,age_gender) for a in img_list)
    all_images=pd.concat(rows)

    all_images.to_hdf(network_dir+'FaceDatabase.h5', 'index', 'w',complevel=9)    

    print("Face images stored in:", network_dir)

    return all_images


def makelist(extension, source_dir):
    templist=[]
    for subdir, dirs, files in os.walk(source_dir):
        dirs[:] = [d for d in dirs if d not in 'Faces']
        for file in files:
            if extension in os.path.join(subdir, file):
                f=os.path.join(subdir, file)
                templist.append(f)
    return templist


def crop_face(image_path, face_dir, age_gender=False, exif=False):
    
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
    faces = np.empty((len(detected), img_size, img_size, 3))


    rows=pd.DataFrame()

    if len(detected) > 0:
        for i, d in enumerate(detected):

            face_img_name="face{}_{}".format(str(i+1), img_name)
            
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            
            crop_face=img[yw1:yw2 + 1, xw1:xw2 + 1]
            encoding = face_recognition.face_encodings(crop_face)
            
            faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size), interpolation = cv2.INTER_AREA)

            if len(encoding)==0:
                break

            if age_gender:
                # predict ages and genders of the detected faces
                results = model.predict(faces)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()

                age=int(predicted_ages[i])
                gender=predicted_genders[i][0]

                img_row = dict({
                    'img_path':image_path,
                    'img_name':img_name,
                    'face_name':face_img_name,
                    'encoding': encoding[0],
                    'age':age,
                    'gender':gender
                    }) 
            else:
                img_row = dict({
                    'img_path':image_path,
                    'img_name':img_name,
                    'face_name':face_img_name,
                    'encoding': encoding[0]
                    }) 
            rows=rows.append(img_row, ignore_index=True)
            cv2.imwrite(face_dir+face_img_name, crop_face)

        return rows


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



def cluster(source_dir, algorithm='DBSCAN', initial_eps=0.44, iterations=1, max_distance=50, mosaic=True):
    '''
    This function uses a density-based clustering algorithm (DBSCAN) to identify clusters of similar faces in the list of facial encodings.
    Starting with loose clustering parameters, the function iteratively increases the number of minimum samples and decreases the neighborhood distance parameter.
    In each iteration, facial similarity within clusters is evaluated using cosine distance and recall is calculated.
    Clusters are matched with obituaries, allowing for the identification of individuals across multiple obituaries and individuals who are still alive. 
    Dense clusters are extracted, and sparse clusters are assigned to be re-evaluated in the next iteration.
    When an iteration returns no new clusters, a final iteration is conducted on the remaining facial encodings using the OPTICS algorithm.
    This enables the identification of clusters with varying densities. 
    The function returns a dataframe containing facial encodings grouped into clusters based on similarity, matched with militant IDs where relevant.
    '''
    from sklearn.cluster import DBSCAN
    from sklearn.cluster import OPTICS
    from sklearn.cluster import AgglomerativeClustering

    global network_dir, face_db, cluster_dir, output_dir

    output_dir=os.path.join(source_dir, "Face Network/")
    network_dir=os.path.join(output_dir, "Data/")
    face_db=pd.read_hdf(network_dir+"FaceDatabase.h5")
    cluster_dir=os.path.join(output_dir, "Clusters/")
    face_dir=os.path.join(output_dir, "Faces/")

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
    from sklearn import preprocessing
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
        cpus=joblib.cpu_count()-1
        rows=Parallel(n_jobs=cpus)(delayed(build_mosaic)(cluster,final_results,face_dir,cluster_dir) for cluster in clusters)

    return final_results




def network(source_dir, scale=10):
    
    from pyvis.network import Network

    global network_dir, face_db, face_dir, output_dir

    output_dir=os.path.join(source_dir, "Face Network/")
    face_dir=os.path.join(output_dir, "Faces/")
    network_dir=os.path.join(output_dir, "Data/")
    face_db=pd.read_hdf(network_dir+"FaceDatabase.h5")
    
 
    #discard faces that were not matched into clusters
    face_db=face_db[face_db['cluster']>0]

    #group the dataframe by image, creating lists of faces in each image
    images=face_db.groupby('img_name')['cluster'].apply(list).reset_index().rename(columns={'cluster':'connections'})
    face_db=pd.merge(face_db,images,how='left',on='img_name')

    #group the dataframe by cluster, creating lists of co-appearances with other clusters
    clusters=face_db.groupby('cluster')['connections'].apply(tuple).reset_index()
    clusters['connections']=clusters['connections'].apply(lambda x: list([item for sublist in x for item in sublist]))
    info=face_db.groupby('cluster').first().reset_index().drop(columns=['connections'])
    exp=clusters.explode(column='connections')
    exp=exp.merge(info, how='left',on='cluster')    

    exp['edge_count']=1
    exp['edge']=exp.apply(lambda x: tuple([x['cluster'], x['connections']]),axis=1)
    exp['total_connections']=np.where(exp['cluster']!=exp['connections'], 1,0)

    weight=exp.groupby('edge')['edge_count'].count().reset_index().rename(columns={'edge_count':'weight'})
    size=exp[['cluster','edge_count','total_connections']].groupby('cluster').agg({'edge_count':'count', 'total_connections':'sum'}).reset_index().rename(columns={'edge_count':'size'})

    exp=exp.drop_duplicates(subset=['edge'], keep='first').drop(columns=['total_connections'])

    exp=pd.merge(exp,weight,on='edge',how='left')
    exp=pd.merge(exp,size,on='cluster',how='left').sort_values(by='total_connections',ascending=False)

    net = Network(height='100%', width='100%')
    net.show_buttons()
    #net.barnes_hut(spring_length=200)
    #net.enable_physics(False)

    for index, row in exp.iterrows():
        src=str(row['cluster'])
        s=np.log10(row['size'])*scale
        connections=str(row['total_connections'])+'<br>'
        image_count=str(int(row['count']))+'<br>'
        path=face_dir+str(row['face_name'])
        
        tag=("Individual ID: "+src+'<br> Connections: '+connections+'Images: '+image_count)

        net.add_node(src, label=src,size=s, title=tag, shape='circularImage',image=path, borderWidth=4)


    for index, row in exp.iterrows():
        src = str(row['cluster'])
        dst = str(row['connections'])
        w = row['weight']


        if src !=dst:
            net.add_edge(src, dst, value=w, title=w)

    neighbor_map = net.get_adj_list()

    net.show(network_dir+'Image_Network.html')
    print("Network graph created in: "+network_dir+'Image_Network.html')
    

def build_mosaic(cluster, df, face_dir, cluster_dir):
    from imutils import build_montages

    image_list=df[df['cluster']==cluster].sort_values(by='cluster_distance_core')['face_name']
    faces=[]
    for i in image_list:
        image = cv2.imread(face_dir+i)
        faces.append(image)
    dim=int(np.sqrt(len(image_list)))+1
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


def accuracy_assessment(df, ID_var, min_count=5):
    from random import random
    from sklearn import metrics
    import matplotlib.pyplot as plt
    import seaborn as sn

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
    df=df[df['ID_count']>min_count]

    clusters=df.groupby('cluster').agg({'ID':lambda x: x.value_counts().index[0]}).reset_index().rename(columns={'ID':'ID_mode'})
    df=pd.merge(df,clusters, how='left',on='cluster')

    from sklearn.metrics.cluster import adjusted_mutual_info_score
    from sklearn.metrics import f1_score

    matched=(len(df[df['cluster']>0])/len(df))
    nmi=adjusted_mutual_info_score(df['ID'], df['cluster'])
    rand=metrics.adjusted_rand_score(df['ID'], df['cluster'])
    homogeneity=metrics.homogeneity_score(df['ID'], df['cluster'])

    contingency_matrix = metrics.cluster.contingency_matrix(df['cluster'], df['cluster_mode'])

    #fig = plt.figure()
    #fig.set_aspect(1)
    #plt.clf()
    #res = sn.heatmap(contingency_matrix, vmax=10, cmap='Blues')

    #plt.show()


    print("% Matched: {}, NMI: {}, Rand: {}, Homogeneity:{}".format(matched, nmi, rand, homogeneity))
    return rand, nmi, homogeneity


def plot_accuracy(photo_dir, bounds=[0,100], xlabel='', **kwargs):
    import matplotlib.pyplot as plt

    accuracy=pd.DataFrame(columns={'max_distance','Matched Images', 'Adjusted Mutual Information','Homogeneity'})

    for i in range(bounds[0],bounds[1]):

        if kwargs['algorithm']=='DBSCAN':
            eps=i/100
        if kwargs['algorithm']=='AHC':
            eps=i/10

        clustered=network(photo_dir, max_distance=i, **kwargs)

        df=pd.read_hdf(photo_dir+'/Face Network/Data/FaceDatabase.h5')

        df['ID']=df['face_name'].str.split('_').str[1:-1]
        df['ID']=['_'.join(map(str, l)) for l in df['ID']]

        metrics=accuracy_assessment(df, df['ID'])

        row=dict({'max_distance':i, 'Rand Index':metrics[0], 'Adjusted Mutual Information':metrics[1], 'Homogeneity':metrics[2]})
        accuracy=accuracy.append(row, ignore_index=True)

        plt.clf()
        plt.plot(accuracy['max_distance'], accuracy['Adjusted Mutual Information'], label='Adjusted Mutual Information')
        plt.plot(accuracy['max_distance'],accuracy['Homogeneity'], label='Homogeneity')
        plt.plot(accuracy['max_distance'],accuracy['Rand Index'], label='Rand Index')
        plt.xlabel(xlabel)
        plt.legend()
    plt.show()
    return plt






