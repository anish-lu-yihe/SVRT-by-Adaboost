
import os
import numpy as np
import csv
#import cv2


def load_svrt_parsing(problem_no):
    X = []
    y = []

    dirs = './data/parsed_classic'
    prbm = 'problem_{:0>2d}'.format(problem_no)
    print('loading ',prbm)
    for img_index in range(10000):
        if img_index % 1000 < 500:
            ans = 1
        else:
            ans = 0
        filename = os.path.join(dirs,prbm,'class_{}'.format(ans),'img_{:0>7d}.txt'.format(img_index))

        with open(filename,'r') as f:
            parsing = csv.reader(f)
            prs = []
            for row in parsing:
                for i in row:
                    prs.append(float(i.strip('Shape()borders()contains()')))

        X.append(prs)
        y.append(ans)

    return (X,y)

def load_svrt_image(problem_no): #this is currently a copy from SVRT-by-RN; the entire load_svrt should be an independent API
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
