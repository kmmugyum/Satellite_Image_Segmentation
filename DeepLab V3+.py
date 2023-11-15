import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

from tensorflow.keras.layers import *
from tensorflow.keras.models import *

#======Layers=========#
class ASPP_Layer(tf.keras.layers.Layer):
    def __init__(self, output_filters=256,kernel_size=3, strides=1, r=6, **kwargs):
        super(ASPP_Layer, self).__init__(**kwargs)
        
        self.conv1=Conv_block(1, use_bias=True)
        
        self.dil_conv_1=Conv_block(1, dilation_rate=1)
        self.dil_conv_6=Conv_block(kernel_size, dilation_rate=4) 
        self.dil_conv_12=Conv_block(kernel_size, dilation_rate=8) 
        self.dil_conv_18=Conv_block(kernel_size, dilation_rate=16)
        
        self.conv_out=Conv_block(filters=output_filters, kernel_size=1)
        
    def build(self, input_shape):
        self.dims=input_shape
        
    def call(self, inputs):
        X=inputs
        X=MaxPool2D()(X)
        X=self.conv1(X)
        # print(self.dims[1]//X.shape[1], self.dims[2]//X.shape[2])
        out_pool=UpSampling2D(
            interpolation='bilinear'
        )(X)
        # print(tf.shape(out_pool))
        out_1=self.dil_conv_1(inputs)
        out_6=self.dil_conv_6(inputs)
        out_12=self.dil_conv_12(inputs)
        out_18=self.dil_conv_18(inputs)
        # print(out_pool.shape, out_1.shape, out_6.shape, out_12.shape, out_18.shape)
        X=Concatenate()([out_pool, out_1, out_6, out_12, out_18])
        output=self.conv_out(X)
        return output
        
    
class Conv_block(tf.keras.layers.Layer):
    def __init__(self
                 , filters=256
                 , kernel_size=3
                 , dilation_rate=1
                 , padding='same'
                 , strides=1
                 , use_bias=False):
        super(Conv_block, self).__init__()
        self.conv1=Conv2D(filters, kernel_size, dilation_rate=dilation_rate, strides=strides, padding=padding, use_bias=use_bias, kernel_initializer='he_normal')
        self.bn1=BatchNormalization()
        self.act=ReLU()
    
    def call(self, inputs):
        X=self.conv1(inputs)
        X=self.bn1(X)
        X=self.act(X)
        return X
    
class Separable_Conv2D(tf.keras.layers.Layer):
    def __init__(self
                , filters
                , kernel_size=3
                , dilation_rate=1
                , strides=1
                , padding='same'
                , use_bias=False
                , **kwargs):
        super(Separable_Conv2D, self).__init__(**kwargs)
        self.sep_conv=SeparableConv2D(filters, kernel_size, dilation_rate=dilation_rate, padding=padding, strides=strides, use_bias=use_bias, kernel_initializer='he_normal')
        self.bn1=BatchNormalization()
        self.act=ReLU()
        
    def call(self, inputs):
        X=self.act(inputs)
        X=self.bn1(X)
        X=self.sep_conv(X)
        return X

    
def DeeplabV3_plus(aspp_flow_cnt=1):
    #======Entry Flow======#
    # Block 1
    INPUT_SHAPE=(512,512,3)
    inputs=Input(shape=INPUT_SHAPE)
    out=Conv_block(32, 3, strides=2, padding='same')(inputs)
    out=Conv_block(64, 3, padding='same')(out)
    residual=Conv2D(128, 1, strides=2, padding='same')(out)
    residual=BatchNormalization()(residual)

    # Block 2
    out=Separable_Conv2D(128, 3, padding='same')(out)
    out=Separable_Conv2D(128, 3, padding='same')(out)
    out=Separable_Conv2D(128, strides=2, padding='same')(out)
    out=Add()([out, residual])
    residual=Conv2D(256, 1, strides=2, padding='same')(out)
    residual=BatchNormalization()(residual)
    skip_con=out

    # Block 3
    out=Separable_Conv2D(256, 3, padding='same')(out)
    out=Separable_Conv2D(256, 3, padding='same')(out)
    out=Separable_Conv2D(256, strides=2, padding='same')(out)
    out=Add()([out, residual])
    residual=Conv2D(728, 1, strides=2)(out)
    residual=BatchNormalization()(residual)

    # Block 4
    out=Separable_Conv2D(728, 3, padding='same')(out)
    out=Separable_Conv2D(728, 3, padding='same')(out)
    out=Separable_Conv2D(728, 3, strides=2, padding='same')(out)
    out=Add()([out, residual])

    for _ in range(16): # embedding layers
        residual=out
        out=ReLU()(out)
        out=BatchNormalization()(out)
        out=SeparableConv2D(728, 3, padding='same')(out)
        out=ReLU()(out)
        out=BatchNormalization()(out)
        out=SeparableConv2D(728, 3, padding='same')(out)
        out=ReLU()(out)
        out=BatchNormalization()(out)
        out=SeparableConv2D(728, 3, padding='same')(out)
        out=Add()([out, residual])

    for _ in range(aspp_flow_cnt):
        out=ASPP_Layer()(out)

    out=UpSampling2D(interpolation='bilinear')(out)
    out=UpSampling2D(interpolation='bilinear')(out)
    out=Concatenate()([out, skip_con])
    out=Conv_block(256)(out)
    out=Conv_block(128)(out)
    out=UpSampling2D(interpolation='bilinear')(out)
    out=UpSampling2D(interpolation='bilinear')(out)
    out=Conv2D(3, 1, activation='softmax')(out)

    model=Model(inputs, out)