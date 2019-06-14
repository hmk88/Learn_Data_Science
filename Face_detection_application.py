# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:52:57 2019

@author: hhaq
"""

#####https://jakevdp.github.io/PythonDataScienceHandbook/
###Face detection application 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

####Histogram of Gradient (HOG) is used for feature extraction
##It was developed to identify pedesttrians within image
from skimage import data, color, feature
import skimage.data

image = color.rgb2gray(data.chelsea())
hog_vec, hog_vis = feature.hog(image, visualise=True)

fig, ax = plt.subplots(1, 2, figsize=(12, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('input image')

ax[1].imshow(hog_vis)
ax[1].set_title('visualization of HOG features');


###########Steps for face detection application 
# 1-Obtain a set of images thumbnails of faces to constitute positive training samples
# 2-Obtain a set of images thumbnails of non-faces to constitute negative training samples
# 3-Extrac HOG features from these training samples 
# 4-Train a linear SVM classifier on these images 
# 5-For an unknown image, pass a sliding window across the image, using the 
# model to evaluate whether that window contains a face or not 
# 6-If detection overlaps, combine them into a single window 
#############################################################



#######  1- Obtaining a set of positive training samples
###download wild dataset from sci-kit learn
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people()
positive_patches = faces.images
positive_patches.shape
#This gives us 13000 face images to use for training 

########  2- Obtaining a set of negative training samples
#creating sets of thumbnail which donot contain faces in them
from skimage import data, transform
imgs_to_use = ['camera', 'text', 'coins', 'moon',
               'page', 'clock', 'immunohistochemistry',
               'chelsea', 'coffee', 'hubble_deep_field']
images = [color.rgb2gray(getattr(data, name)())
          for name in imgs_to_use]

from sklearn.feature_extraction.image import PatchExtractor
def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size,
                               max_patches=N, random_state=0)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size)
                            for patch in patches])
    return patches

negative_patches = np.vstack([extract_patches(im, 1000, scale)
                              for im in images for scale in [0.5, 1.0, 2.0]])
negative_patches.shape
##########Now we have 30000 suitable image patches which donot contain faces. Let's take a look at few of them
fig, ax = plt.subplots(6, 10)
for i, axi in enumerate(ax.flat):
    axi.imshow(negative_patches[500 * i], cmap='gray')
    axi.axis('off')
#Our hope is that these would cover the space of non-faces sufficiently that the algorithm is going to see
    
#   3- Combine sets and extract HOG features
from itertools import chain
X_train = np.array([feature.hog(im)
                    for im in chain(positive_patches,
                                    negative_patches)])
y_train = np.zeros(X_train.shape[0])
y_train[:positive_patches.shape[0]] = 1

X_train.shape
########We are left with 43000 training samples in 1215 dimensions.
####Now we have the data to feed into sci-kit learn



#  4- Training a support vector mahine (SVM)
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
cross_val_score(GaussianNB(), X_train, y_train)

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]})
grid.fit(X_train, y_train)
grid.best_score_

grid.best_params_

model = grid.best_estimator_
model.fit(X_train, y_train)



#  5- Find faces in a new image
#We will take one portion of astronaut image, run sliding window over it and evaluate each patch 

test_image = skimage.data.astronaut()
test_image = skimage.color.rgb2gray(test_image)
test_image = skimage.transform.rescale(test_image, 0.5)
test_image = test_image[:160, 40:180]

plt.imshow(test_image, cmap='gray')
plt.axis('off');

######Creating window that iterates over patches of this image and compute HOG features for each patch 
def sliding_window(img, patch_size=positive_patches[0].shape,
                   istep=2, jstep=2, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch
            
indices, patches = zip(*sliding_window(test_image))
patches_hog = np.array([feature.hog(patch) for patch in patches])
patches_hog.shape

###Finally, using the model to evaluate whether each patch contains a face
labels = model.predict(patches_hog)
labels.sum()
#########out of 2000 patches, we have found 30 or so detections
#Using this information about these patches to show where they lie on the test image, drawing them as rectangles
fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')
ax.axis('off')

Ni, Nj = positive_patches[0].shape
indices = np.array(indices)

for i, j in indices[labels == 1]:
    ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',
                               alpha=0.3, lw=2, facecolor='none'))













