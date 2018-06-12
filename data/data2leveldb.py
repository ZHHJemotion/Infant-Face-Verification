# -*- coding: utf-8 -*-
"""
    Created on Tue Mar 21 2017

    @author: ZHHJemotion

    Using this code we can get the train/test set with the ratio of negative and positive pairs is 1:1

"""

import numpy as np
import deepdish as dd
import glob
import random
from PIL import Image
import plyvel
import caffe

# from pandas import Series,DataFrame
# import pandas as pd

# !!! By changing the path to get the train/test set
datapath = '/Users/zhangxingjian/Desktop/Programming/Python/Conver_Data/TrainingSet/*.png'
# datapath = '/Users/zhangxingjian/Desktop/Programming/Python/Conver_Data/TestSet/*.png'

files = glob.glob(datapath)
sortedfiles = sorted(files)
nrfiles = np.size(sortedfiles)
N = 640  # training data (positive : negative = 1 : 1)!!!! can be changed

# Let's pretend this is interesting data
X = np.zeros((N, 3, 160, 160), dtype=np.uint8)

# We need to prepare the database for the size. If you don't have
# deepdish installed, just set this to something comfortably big
# (there is little drawback to settings this comfortably big).
map_size = dd.bytesize(X) * 2.5
db = plyvel.DB('testLDB_FGNet_640_aligned_cleaned/', create_if_missing=True, error_if_exists=True,
               write_buffer_size=map_size)  # trainLDB_40kids_128000_aligned_balanced_cv10
wb = db.write_batch()

count = 0
count_notsame = 0
count_same = 0
count_0 = 0  # the number of the negative pairs
count_1 = 0  # the number of the positive pairs

run = True

while run:
    for k in range(N):

        if count == 640:  # be changed !!!!
            run = False
            break

        i = random.randrange(nrfiles)
        j = random.randrange(nrfiles)            # 70-74 to train data(all kids), 83-85 for test set(FG-Net)
        label_i = np.str(sortedfiles[i][83:85])  # 69-73 to test data, 73-77 to training data !!!!! need to be changed
        label_j = np.str(sortedfiles[j][83:85])  # For 2 kids, 77-81 for test, 78-82 for training
        imageName_i = np.str(sortedfiles[i][:])  # For 4,6,8 kids, 84-88 for test, 88-92 for training
        imageName_j = np.str(sortedfiles[j][:])  # For 10 kids, 85-89 for test, 89-93 for training
                                                 # For all kids independent, 76-80 for test, 77-81 for training
        if imageName_i == imageName_j:
            break

        else:
            if label_i == label_j:
                label = 1
                count_same += 1

            else:
                label = 0
                count_notsame += 1

            if count_same > 1:
                count_same = 1
                break
            elif count_notsame > 1:
                count_notsame = 1
                break

            else:
                img_1 = Image.open(sortedfiles[i])
                img_1 = img_1.resize((128, 128), Image.BILINEAR)
                img1 = np.float64(img_1)
                img1 = img1[:, :, (2, 1, 0)]
                img1 = img1.transpose((2, 0, 1))

                img_2 = Image.open(sortedfiles[j])
                img_2 = img_2.resize((128, 128), Image.BILINEAR)
                img2 = np.float64(img_2)
                img2 = img2[:, :, (2, 1, 0)]
                img2 = img2.transpose((2, 0, 1))

                img3 = np.concatenate((img1, img2))

                datum = caffe.io.array_to_datum(img3, label)

                wb.put('%08d' % count, datum.SerializeToString())
                count += 1

                if count % 128 == 0:  # 64 for training, 100 for test !!!!!
                    # Write batch of images to database
                    wb.write()
                    del wb
                    wb = db.write_batch()
                    print 'Processed %i pairs.' % count

                if count % 640 != 0:  # change !!!!!!!
                    # Write last batch of images
                    wb.write()

                if count_notsame == 1 and count_same == 1:
                    count_notsame = 0
                    count_same = 0
                    count_0 += 1
                    count_1 += 1

                if count >= 640:  # !!!!!! be changed
                    run = False
                    break

print count_1, count_0