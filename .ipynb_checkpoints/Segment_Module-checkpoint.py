from DeepLab_V3_plus import DeeplabV3_plus
from MaskToPolygon import MaskToPolygon
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class Segment_Module:
    loaded = False
    def __init__(self, option, epsilon):
        model = DeeplabV3_plus()
        self.MTP   = MaskToPolygon(epsilon)
        
        model.load_weights('../models_2term/[CR] TFRecord_aug_1.h5')
        self.model = model
        self.option = option
    
    def __call__(self, img):
        if np.max(img) > 1:
            img = img / 255.
        pred = np.array(self.model(img[np.newaxis,...]))
        pred = self.MTP(pred[0], self.option)
        return pred

class SingleTon:
    _instance = None
    prev_epsil = None
    prev_option = None
    
    def __new__(cls, option, epsilon):  
        if not cls._instance or cls.prev_epsil != epsilon or cls.prev_option != option:
            cls.prev_epsil = epsilon
            cls._instance = Segment_Module(option, epsilon)
        return cls._instance