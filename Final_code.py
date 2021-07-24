# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 07:34:34 2019

@author: Shahil
"""

#Import required libraries
import cv2
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import re
import glob
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.datasets import mnist

## for Model definition/training
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, concatenate,  Dropout
from keras.optimizers import Adam

from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

## required for semi-hard triplet loss:
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf


os.chdir('D:/Thesis')
#Load the weights of OpenPose

protoFile = "D:/Thesis/pose_deploy_linevec.prototxt"
weightsFile = "D:/Thesis/pose_iter_440000.caffemodel"
nPoints = 18
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]


#Initialize Parameters

inWidth = 256
inHeight = 256
threshold = 0.1

#Function for angle calcualtion 
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

#Get the list of videos in to a file 
flist=glob.glob(r'D:\Thesis\al902\*')
#input_source = flist[0]
input_source = r'D:\Thesis\sam2.avi'

#Read video frame by frame using opencv videocapture
cap = cv2.VideoCapture(input_source)
cap.set(cv2.CAP_PROP_POS_FRAMES, 40)
hasFrame, frame = cap.read()
vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

#load the COCO OpenCV weights
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

#Run OpenPose Video captue for every frame in every and store the co-ordinate as list. Write the coordinate included frames in a video

pot=[]
n=371
nin=[]
while cv2.waitKey(1) < 0:
    t = time.time()
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
        pot.append(input_source)
        nin.append(pot)
        n=n+1
        if n>371:
            cv2.waitKey()
            break
        input_source = flist[n]
        print(input_source)
        cap = cv2.VideoCapture(input_source)
        hasFrame, frame = cap.read()
        pot=[]
        #with open("tmp.txt", "wb") as fp:
         #   pickle.dump(nin, fp)
    
        #cv2.waitKey()
        #break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append((None,None))
    #Calculate different angles    
    if points[11][0] and points[9][0] and points[12][0]:
            ang=getAngle(points[9], points[11], points[12])
            if ang>180:
                ang=360-ang
            
            points.append(ang) 
    else:
        points.append(None)
    if points[11][0] and points[5][0] and points[6][0]:
        
            ang=getAngle(points[11], points[5], points[6])
            if ang>180:
                ang=360-ang
            
            points.append(ang) 
    else:
        points.append(None)
    if points[11][0] and points[1][0]:
        p1=points[11]
        p2=points[1]
        l=math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
            
        points.append(l) 
    else:
        points.append(None)
    if points[11][0] and points[12][0]:
        p1=points[11]
        p2=points[12]
        l=math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
        points.append(l) 
    else:
        points.append(None)
    if points[5][0] and points[6][0]:
        p1=points[5]
        p2=points[6]
        l=math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
            
        points.append(l) 
    else:
        points.append(None)
    pot.append(points)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA][0] and points[partB][0]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.ellipse(frame, points[partA], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.ellipse(frame, points[partB], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, str(partA), points[partA], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),2,cv2.LINE_AA)
            cv2.putText(frame, str(partB), points[partB], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),2,cv2.LINE_AA)
            ###cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-2, lineType=cv2.FILLED)
            #cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-2, lineType=cv2.FILLED)

    cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Output-Keypoints', frameCopy)
    cv2.imshow('Output-Skeleton', frame)
    

    vid_writer.write(frame)

cv2.waitKey()
vid_writer.release()


#Dump the coordinate list in a pickle file and load when required
with open("full.txt", "wb") as fp:
    pickle.dump(nin, fp)#Pickling
with open("full3.txt", "rb") as fp: 
    b = pickle.load(fp)
    
pot=b[2]

#Calculate various manual features using the cordinates for every video and save as a csv 
j=0
for potf in b:
    print(potf[-1])
    p1=[potf[x][1][1]for x in range(0,(len(potf)-1))]
    p1=[int(x) for x in p1 if x is not None]
    p1s=pd.Series(p1).value_counts()
    p1m=p1s.index[0]
    p1d=np.median(p1)
    #p1d=p1s.index[0]-p1s.index[1]
    
    p11=[potf[x][11][1]for x in range(0,(len(potf)-1))]
    p11=[int(x) for x in p11 if x is not None]
    p11s=pd.Series(p11).value_counts()
    p11m=p11s.index[0]
    p11d=np.median(p11)
    #if len(p11s) >1 :
        #p11d=p11s.index[0]-p11s.index[1]
    #else :
    #    p11d=0
    
    
    p12=[potf[x][12][1]for x in range(0,(len(potf)-1))]
    p12=[int(x) for x in p12 if x is not None]
    p12s=pd.Series(p12).value_counts()
    p12m=p1s.index[0]
    p12d=np.median(p12)
    #p12d=p12s.index[0]-p12s.index[1]
    
    p13=[potf[x][13][1]for x in range(0,(len(potf)-1))]
    p13=[int(x) for x in p13 if x is not None]
    p13s=pd.Series(p13).value_counts()
    p13m=p13s.index[0]
    p13d=np.median(p13)
    #p13d=p13s.index[0]-p13s.index[1]
    
    p7=[potf[x][7][1]for x in range(0,(len(potf)-1))]
    p7=[int(x) for x in p7 if x is not None]
    p7s=pd.Series(p7).value_counts()
    p7m=p7s.index[0]
    p7d=np.median(p7)
    #if len(p7s)>1:
    #    p7d=p7s.index[0]-p7s.index[1]
    #else:
    #    p7d=0
    
    
    p18=[potf[x][18]for x in range(0,(len(potf)-1))]
    p18=[int(x) for x in p18 if x is not None]
    p18s=pd.Series(p18).value_counts()
    p18m=p18s.index[0]
    p18d=np.median(p18)
    #p18d=p18s.index[0]-p18s.index[1]
    
    p19=[potf[x][19]for x in range(0,(len(potf)-1))]
    p19=[int(x) for x in p19 if x is not None]
    p19s=pd.Series(p19).value_counts()
    p19m=p19s.index[0]
    p19d=np.median(p19)
    #p19d=p19s.index[0]-p19s.index[1]
    
    p20=[potf[x][20]for x in range(0,(len(potf)-1))]
    p20=[int(x) for x in p20 if x is not None]
    p20s=pd.Series(p20).value_counts()
    p20m=p20s.index[0]
    p20d=np.median(p20)
    #if len(p20s)>1:
    #    p20d=p20s.index[0]-p20s.index[1]
    #else:
    #    p20d=0
    
    
    p21=[potf[x][21]for x in range(0,(len(potf)-1))]
    p21=[int(x) for x in p21 if x is not None]
    p21s=pd.Series(p21).value_counts()
    p21m=p21s.index[0]
    p21d=np.median(p21)
    #p21d=p21s.index[0]-p21s.index[1]
    
    p22=[potf[x][22]for x in range(0,(len(potf)-1))]
    p22=[int(x) for x in p22 if x is not None]
    p22s=pd.Series(p22).value_counts()
    p22m=p22s.index[0]
    p22d=np.median(p22)
    #p22d=p22s.index[0]-p22s.index[1]
    
    pnam=str(potf[-1])[16:19]#[18:20]
    
    l1=[[p1m,p1d,p11m,p11d,p12m,p12d,p13m,p13d,p7m,p7d,p18m,p18d,p19m,p19d,p20m,p20d,p21m,p21d,p22m,p22d,pnam]]
    if j==0:
        fin=pd.DataFrame(l1)
        j=1
    else:
        fin=fin.append(l1)
    
fin.columns=['p1m','p1d','p11m','p11d','p12m','p12d','p13m','p13d','p7m','p7d','p18m','p18d','p19m','p19d','p20m','p20d','p21m','p21d','p22m','p22d','pnam']

#fin.to_csv('finlist.csv',index=False)

fin=pd.read_csv('finlist.csv')

##Knn

# Subset the traing dat for KNN

X = fin.iloc[:54, :-1].values
y = fin.iloc[:54, 20].values

#X = preprocessing.normalize(X)
#pca = PCA(n_components=10)
#X = pca.fit_transform(X)

#Split in to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.3,random_state=15)

from sklearn.neighbors import KNeighborsClassifier
# Calculating error for various K values to chose the optimum Kvalue
error = []
for i in range(3, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    
plt.rc('axes', labelsize=18)
plt.figure(figsize=(12, 6))
plt.plot(range(3, 10), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value(triplet loss)')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
plt.savefig('img9.png')

#Build Knn with optimum K value and print the classification report

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

plt.rc('axes', labelsize=18)


###TRiplet Loss
l=[None in x for x in b]
new=[]

for i in b:
    #nt=[]
    l=int(len(i)/2)
    #t1=i[60:90]
    t1=i[(l-15):(l+15)]
    new.append(t1)
new2=[]        
for i in new:
    t2=[x[0:14] for x in i]
    new2.append(t2)
    
    
    
new3=np.array(new2)
new3=new3.reshape(744*30*28)
new4=pd.DataFrame(new3)
new4=new4.fillna(85)
new5=np.array(new4)
new7=new5.reshape(744,30,28)

######Triplet loss###################

## dataset
from keras.datasets import mnist

## for Model definition/training
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, concatenate,  Dropout
from keras.optimizers import Adam

from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

## required for semi-hard triplet loss:
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf

## for visualizing 
import matplotlib.pyplot as plt, numpy as np
from sklearn.decomposition import PCA

def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.

    output[i, j] = || feature[i, :] - feature[j, :] ||_2

    Args:
      feature: 2-D Tensor of size [number of data, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.

    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
        math_ops.reduce_sum(
            math_ops.square(array_ops.transpose(feature)),
            axis=[0],
            keepdims=True)) - 2.0 * math_ops.matmul(feature,
                                                    array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances

def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.

    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(data - axis_minimums, mask), dim,
        keepdims=True) + axis_minimums
    return masked_maximums

def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.

    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(data - axis_maximums, mask), dim,
        keepdims=True) + axis_maximums
    return masked_minimums

def triplet_loss_adapted_from_tf(y_true, y_pred):
    del y_true
    margin = 1.
    labels = y_pred[:, :1]

 
    labels = tf.cast(labels, dtype='int32')

    embeddings = y_pred[:, 1:]

    ### Code from Tensorflow function [tf.contrib.losses.metric_learning.triplet_semihard_loss] starts here:
    
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    # lshape=array_ops.shape(labels)
    # assert lshape.shape == 1
    # labels = array_ops.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pdist_matrix = pairwise_distance(embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    # global batch_size  
    batch_size = array_ops.size(labels) # was 'array_ops.size(labels)'

    # Compute the mask.
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(
        array_ops.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(
            pdist_matrix_tile, array_ops.reshape(
                array_ops.transpose(pdist_matrix), [-1, 1])))
    mask_final = array_ops.reshape(
        math_ops.greater(
            math_ops.reduce_sum(
                math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
            0.0), [batch_size, batch_size])
    mask_final = array_ops.transpose(mask_final)

    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    mask = math_ops.cast(mask, dtype=dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = array_ops.reshape(
        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = array_ops.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = array_ops.tile(
        masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = array_ops.where(
        mask_final, negatives_outside, negatives_inside)

    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = math_ops.cast(
        adjacency, dtype=dtypes.float32) - array_ops.diag(
        array_ops.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    semi_hard_triplet_loss_distance = math_ops.truediv(
        math_ops.reduce_sum(
            math_ops.maximum(
                math_ops.multiply(loss_mat, mask_positives), 0.0)),
        num_positives,
        name='triplet_semihard_loss')
    
    ### Code from Tensorflow function semi-hard triplet loss ENDS here.
    return semi_hard_triplet_loss_distance

def create_base_network(image_input_shape, embedding_size):
    """
    Base network to be shared (eq. to feature extraction).
    """
    input_image = Input(shape=image_input_shape)

    x = Flatten()(input_image)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(embedding_size)(x)

    base_network = Model(inputs=input_image, outputs=x)
    plot_model(base_network, to_file='base_network.png', show_shapes=True, show_layer_names=True)
    return base_network

#X = fin.iloc[:, :-1].values
y = fin.iloc[:, 20].values


if __name__ == "__main__":
    # in case this scriot is called from another file, let's make sure it doesn't start training the network...

    batch_size = 128
    epochs = 10000
    train_flag = True  # either     True or False

    embedding_size = 64

    no_of_components = 2  # for visualization -> PCA.fit_transform()

    step = 10

    # The data, split between train and test sets
    
    x_train, x_test, y_train, y_test = train_test_split(new7, y,stratify=y, test_size=0.20,random_state=26)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 256.
    x_test /= 256.
    input_image_shape = (30, 28, 1)
    x_val = x_train[:50]
    y_val = y_train[:50]

# Network training...
    if train_flag == True:
        base_network = create_base_network(input_image_shape, embedding_size)

        input_images = Input(shape=input_image_shape, name='input_image') # input layer for images
        input_labels = Input(shape=(1,), name='input_label')    # input layer for labels
        embeddings = base_network([input_images])               # output of network -> embeddings
        labels_plus_embeddings = concatenate([input_labels, embeddings])  # concatenating the labels + embeddings

        # Defining a model with inputs (images, labels) and outputs (labels_plus_embeddings)
        model = Model(inputs=[input_images, input_labels],
                      outputs=labels_plus_embeddings)

        model.summary()
        #plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
        
        plot_model(base_network, to_file='base_network.png', show_shapes=True, show_layer_names=True)

        # train session
        opt = Adam(lr=0.0001)  # choose optimiser. RMS is good too!

        model.compile(loss=triplet_loss_adapted_from_tf,
                      optimizer=opt)

        filepath = "trip_loss2_ep{epoch:02d}_BS%d.hdf5" % batch_size
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, period=10000)
        callbacks_list = [checkpoint]

        # Uses 'dummy' embeddings + dummy gt labels. Will be removed as soon as loaded, to free memory
        dummy_gt_train = np.zeros((len(x_train), embedding_size + 1))
        dummy_gt_val = np.zeros((len(x_val), embedding_size + 1))

        x_train = np.reshape(x_train, (len(x_train), x_train.shape[1], x_train.shape[2], 1))
        x_val = np.reshape(x_val, (len(x_val), x_train.shape[1], x_train.shape[2], 1))

        H = model.fit(
            x=[x_train,y_train],
            y=dummy_gt_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([x_val, y_val], dummy_gt_val),
            callbacks=callbacks_list)
        
        plt.figure(figsize=(8,8))
        plt.plot(H.history['loss'], label='Training loss')
        plt.plot(H.history['val_loss'], label='validation loss')
        plt.legend()
        plt.title('Training loss')
        plt.savefig('img8.png')
        plt.show()
    else:

        #####
        model = load_model('trip_loss2_ep10000_BS128.hdf5',
                                        custom_objects={'triplet_loss_adapted_from_tf':triplet_loss_adapted_from_tf})

#Plot the clustering of PCA components of embedding output
x_plot=x_train
y_plot=y_train   
testing_embeddings = create_base_network(input_image_shape,
                                             embedding_size=embedding_size)
x_embeddings_before_train = testing_embeddings.predict(np.reshape(x_plot, (len(x_plot), 30, 28, 1)))
    # Grabbing the weights from the trained network
    for layer_target, layer_source in zip(testing_embeddings.layers, model.layers[2].layers):
        weights = layer_source.get_weights()
        layer_target.set_weights(weights)
        del weights
x_embeddings = testing_embeddings.predict(np.reshape(x_plot, (len(x_plot), 30, 28, 1)))
dict_embeddings = {}
dict_gray = {}
test_class_labels = np.unique(np.array(y_plot))
test_class_labels=test_class_labels[0:20]
pca = PCA(n_components=no_of_components)
decomposed_embeddings = pca.fit_transform(x_embeddings)
#     x_test_reshaped = np.reshape(x_test, (len(x_test), 28 * 28))
decomposed_gray = pca.fit_transform(x_embeddings_before_train)

fig = plt.figure(figsize=(14, 14))
plt.rc('axes', titlesize=20)
for label in test_class_labels:
    decomposed_embeddings_class = decomposed_embeddings[y_plot == label]
    decomposed_gray_class = decomposed_gray[y_plot == label]

    plt.subplot(1,2,1)
    plt.scatter(decomposed_gray_class[:step,1], decomposed_gray_class[:step,0],label=str(label))
    plt.title('Before training (embeddings)')
    plt.legend()

    plt.subplot(1,2,2)
    plt.scatter(decomposed_embeddings_class[:step, 1], decomposed_embeddings_class[:step, 0], label=str(label))
    plt.title('After @%d epochs' % epochs)
    plt.legend()

#plt.show()
plt.savefig('img5.png')


#Knn for triplet algorithm
X = testing_embeddings.predict(np.reshape(new7, (len(new7), 30, 28, 1)))
#X=x_embeddings
y = fin.iloc[:, 20].values

X=X[690:744]
y=y[690:744]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.3,random_state=15)


# Calculating error for various K values to chose the optimum Kvalue
error = []
for i in range(3, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    
plt.rc('axes', labelsize=18)
plt.figure(figsize=(12, 6))
plt.plot(range(3, 10), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value(triplet loss)')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.savefig('img9.png')

#Build Knn with optimum K value and print the classification report
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=4)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

