
# coding: utf-8

# In[14]:

import tensorflow as tf
tf.reset_default_graph()

from PIL import Image  
import matplotlib.pyplot as plt
import numpy as np
import nn_model
from scipy import io
import os

num_test_data = 100
img_W = 8      
img_H = 8    
label_W = 32
label_H = 32
batch_size = 1

num_patterns = img_W * img_H
num_pixels = label_W * label_H

DLGI = np.zeros([label_W,label_H,num_test_data])
isSave = True

data = io.loadmat('DLGI_exp_data_64.mat')

result_save_path = '.\\results\\%d\\' % (num_patterns)
model_save_path = '.\\models\\model_mnist_%d_%d.ckpt'%(num_pixels, num_patterns)

if not os.path.exists(result_save_path):
    os.makedirs(result_save_path) 

inputs = data['y']
labels = data['x']

x = tf.placeholder(tf.float32, shape=[batch_size,num_patterns])
isTrain = tf.placeholder(tf.bool, name = 'isTrain')
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
DLGI_re = nn_model.inference(x, batch_size, isTrain, keep_prob, num_patterns)

saver = tf.train.Saver()

print('Start testing...') 
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    saver.restore(sess,model_save_path)  
    
    for i in range(num_test_data):
        # load input and preprocess       
        test_data = np.reshape(inputs[i,:], [1, num_patterns])
        images_std = np.sqrt(np.sum(np.square(test_data - np.mean(test_data)))/num_patterns)        # CNN_3232_64_bzh.ckpt
        test_data = (test_data-np.mean(test_data))/images_std

        # calculate the corresponding output
        temp = sess.run(DLGI_re,feed_dict={x:test_data,isTrain:False,keep_prob:1.0})
        DLGI[:,:,i] = np.reshape(temp,[label_W,label_H])
        
        if isSave:
            # save to local
            img_o = np.reshape(temp,[label_W, label_H])
            img_o = Image.fromarray(img_o.astype('uint8')).convert('L')
            img_o.save(result_save_path + str(i) + '.bmp')
    print('Finished!')  
    print('results were saved in %s' % result_save_path)
    
    # Visualization
    n = int(np.sqrt(num_test_data))
    count = 0
    figure_preds = np.zeros((label_W * n, label_H *n))
    figure_labels = np.zeros((label_W * n, label_H *n))
    for j in range(n):
        for k in range(n):
            figure_preds[j*label_W:(j+1)*label_W, k*label_H:(k+1)*label_H] = DLGI[:,:,count]
            figure_labels[j*label_W:(j+1)*label_W, k*label_H:(k+1)*label_H] = labels[:,:,count]
            count += 1
    plt.figure(figsize=(15,30))
    plt.subplot(121)
    plt.imshow(figure_preds, cmap='gray')
    plt.title('DLGI exp results')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(figure_labels, cmap='gray')
    plt.title('Labels')
    plt.axis('off')
    plt.rcParams['font.size'] = 20
    plt.show()   
   
     
