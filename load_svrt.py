
import os

import numpy as np
import cv2
import pickle

def load_svrt(problem_no):
    qst = np.zeros(11)
    sample = []

    dirs = './data/svrt/images'
    prbm = 'problem_{:0>2d}'.format(problem_no)
    print('loading ',prbm)
    for img_index in range(10000):
        if img_index % 1000 < 500:
            ans = 1
        else:
            ans = 0
        filename = os.path.join(dirs,prbm,'class_{}'.format(ans),'img_{:0>7d}.png'.format(img_index))

        img = cv2.imread(filename) / 255
        img = cv2.resize(img, (75,75), cv2.INTER_LINEAR)
        img = np.swapaxes(img,0,2)

        sample.append((img,qst,ans))


    rel_train = sample[:9000]
    rel_test = sample[9000:]

    return (rel_train, rel_test)

def load_data():
    print('loading data...')
    dirs = './data'
    filename = os.path.join(dirs,'sort-of-clevr.pickle')
    with open(filename, 'rb') as f:
      train_datasets, test_datasets = pickle.load(f)
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')

    for img, relations, norelations in train_datasets:
        img = np.swapaxes(img,0,2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans))

    for img, relations, norelations in test_datasets:
        img = np.swapaxes(img,0,2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans))

    return (rel_train, rel_test, norel_train, norel_test)
