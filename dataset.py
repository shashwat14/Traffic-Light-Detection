# -*- coding: utf-8 -*-

import csv
import scipy.misc
from keras.utils import np_utils
import numpy as np
class Dataset(object):

    def __init__(self):
        self.location = "C:\\Users\\Shashwat\\Storage\\Datasets\\TrafficLight\\"
        self.train_ptr = 0
        self.val_ptr = 0
        self.urls = []
        self.categories = []
        with open(self.location + 'labels.csv') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                imageLoc = row[0]
                category = int(row[1])
                self.urls.append(self.location + imageLoc)
                self.categories.append(category)
        
        
        size = len(self.urls)
        #one-hot
        self.categories = np_utils.to_categorical(self.categories, 3)
        
        #split data into training an validation
        self.train_x = self.urls[:int(size*0.8)]
        self.train_y = self.categories[:int(size*0.8)]
        self.val_x = self.urls[int(-size*0.2):]
        self.val_y = self.categories[int(-size*0.2):]
        
        self.train_size = len(self.train_x)
        self.val_size = len(self.val_x)
        
    def LoadTrainingBatch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            img = scipy.misc.imread(self.train_x[(self.train_ptr + i)%self.train_size])
            if img.shape != (256,455,3):
                scipy.misc.imrotate(img, 90)
            x_out.append(img.reshape(3,256,455))
            y_out.append(self.train_y[(self.train_ptr + i)%self.train_size])
            
        self.train_ptr += batch_size
        
        return tuple((np.array(x_out), np.array(y_out)))
    
    def LoadValidationBatch(self, batch_size):
        x_out = []
        y_out = []
        for i in range(0, batch_size):
            img = scipy.misc.imread(self.val_x[(self.val_ptr + i)%self.val_size])
            if img.shape != (256,455,3):
                scipy.misc.imrotate(img, 90)
            x_out.append(img.reshape(3,256,455))
            y_out.append(self.val_y[(self.val_ptr + i)%self.val_size])
            
        self.val_ptr += batch_size
        
        return tuple((np.array(x_out), np.array(y_out)))
    
    def LoadSample(self):
        img = scipy.misc.imread(self.train_x[0])
        print (img.shape)