from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import tarfile
#from IPython.display import display, video
#from scipy import ndvideo
import scipy.misc
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import random
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#import tensorflow as tf
import cv2

# Config the matplotlib backend as plotting inline in IPython
#%matplotlib inline
data_root='pickles'
train_filename=''
test_filename=''
train_folders=['new_videos/Abnormal','new_videos/Normal']
PICKLE_FILE='crowd_128_big.pickle'

num_classes = 2
np.random.seed(133)
video_size = 128  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
NUM_CHANNELS=3
NUM_FRAMES= 60

train_size =216          	# num_classes * training videos per class
valid_size =48		   		# num_classes * validation videos per class
test_size =2			# num_classes * test videos per class

def getFrames(videofile):
    vidcap = cv2.VideoCapture(videofile)
    success,image = vidcap.read()
    count = 0
    success = True
    framestack=np.ndarray(shape=(video_size,video_size,NUM_CHANNELS*NUM_FRAMES));
    while count<NUM_FRAMES:
      success,image = vidcap.read()
      print(count)
      print(videofile)
      print(image.size)
      image=cv2.resize(image,(video_size,video_size))
      framestack[:,:,NUM_CHANNELS*count:NUM_CHANNELS*count+NUM_CHANNELS]=image;
      count += 1
    return framestack

def load_video(folder):
    """Load the data for a single video label."""
    video_files = os.listdir(folder)
    random.shuffle(video_files)
    dataset = np.ndarray(shape=(len(video_files), video_size, video_size,NUM_CHANNELS*NUM_FRAMES),dtype=np.float32)
    print(folder)
    num_videos = 0
    for video in video_files:
        video_file = os.path.join(folder,video)

        try:
       	    video_data = getFrames(video_file)
            dataset[num_videos, :, :,:] = video_data
            num_videos = num_videos + 1
        except IOError as e:
            print('Could not read:', video_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_videos, :, :,:]
    dataset=(dataset-np.mean(dataset))/pixel_depth

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
          # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_video(folder)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names

train_datasets = maybe_pickle(train_folders)
"""
pickle_file = train_datasets[0]  # index 0 should be all Beachs, 1 = all Bs, etc.
with open(pickle_file, 'rb') as f:
    video_set = pickle.load(f)  # unpickle
    sample_idx = random.randint(0,len(video_set))  # pick a random video index
    sample_video = video_set[sample_idx, :, :,:]  # extract a 2D slice
    plt.figure()
    plt.imshow(sample_video)  # display it
    plt.show()
"""
def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size,NUM_CHANNELS*NUM_FRAMES), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size,test_size, valid_size):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, video_size)
    test_dataset, test_labels = make_arrays(test_size, video_size)
    train_dataset, train_labels = make_arrays(train_size, video_size)
    vsize_per_class = valid_size // num_classes
    testsize_per_class=test_size//num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t ,start_test= 0, 0, 0
    end_v, end_t ,end_test= vsize_per_class, tsize_per_class,testsize_per_class
    end_l = vsize_per_class+tsize_per_class+testsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                video_set = pickle.load(f)
                # let's shuffle the videos to have random validation and training set
                np.random.shuffle(video_set)
                if valid_dataset is not None:
                    valid_video = video_set[:vsize_per_class, :, :,:]
                    valid_dataset[start_v:end_v, :, :,:] = valid_video
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                test_video = video_set[vsize_per_class:vsize_per_class+testsize_per_class, :, :,:]
                test_dataset[start_test:end_test, :, :,:] = test_video
                test_labels[start_test:end_test] = label
                start_test += testsize_per_class
                end_test += testsize_per_class


                train_video = video_set[vsize_per_class+testsize_per_class:end_l, :, :,:]
                train_dataset[start_t:end_t, :, :,:] = train_video
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels,test_dataset,test_labels, train_dataset, train_labels

valid_dataset, valid_labels,test_dataset,test_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size,test_size, valid_size)
#_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

pickle_file = os.path.join(data_root, PICKLE_FILE)

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
        }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
