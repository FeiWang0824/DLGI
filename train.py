
# coding: utf-8

# In[1]:

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import Input_Data
import nn_model

tf.reset_default_graph()

img_W = 8      
img_H = 8    
label_W = 32
label_H = 32
batch_size = 5 
num_epochs = None 

num_pixels = label_W * label_H
num_patterns = img_W * img_H

train_intensitys_path = '.\\data\\Inputs_mnist_train_64\\'
train_labels_path = '.\\data\\Labels_mnist_train\\'
train_TFRecord_path = '.\\data\\train_minst_64.tfrecord'

model_save_path = '.\\models\\model_mnist_%d_%d.ckpt'%(num_pixels, num_patterns)

# print('Generating TFRecord file...')
# print('This may take some time')
# Input_Data.generate_TFRecordfile(train_intensitys_path,train_labels_path,train_TFRecord_path)
# print('finished!')

with tf.variable_scope('mini-batch'):
    Train_Intensitys_Batch,Train_Labels_Batch = Input_Data.get_batch(train_TFRecord_path, img_W,img_H, label_W,label_H, batch_size,num_epochs)
    
with tf.variable_scope('input'):
    y = tf.placeholder(tf.float32, shape=[None,img_W,img_H],name = 'images')
    x_label = tf.placeholder(tf.float32,shape=[None,label_W,label_H,1],name = 'labels')
    isTrain = tf.placeholder(tf.bool,name = 'isTrain')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

x_conv = nn_model.inference(y, batch_size, isTrain, keep_prob, img_W*img_H)

with tf.variable_scope('loss_function'):
    loss = tf.reduce_mean(tf.square(x_conv - x_label))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)      
with tf.variable_scope('train_step'):
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)

init_op = (tf.local_variables_initializer(),tf.global_variables_initializer())

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator() 
    threads = tf.train.start_queue_runners(sess=sess,coord=coord) 

    print('Start training...')
    try:
        for step in range(500000):
            if coord.should_stop():                
                break
            
            train_intensitys_batch,train_labels_batch = sess.run([Train_Intensitys_Batch,Train_Labels_Batch])
            train_labels_batch = np.reshape(train_labels_batch,[batch_size,label_W,label_H,1])
            
            if step%1000 == 0: 
                train_loss = sess.run(loss,feed_dict={y:train_intensitys_batch,x_label:train_labels_batch,isTrain:True,keep_prob:1.0})
                print('[step %d]: training loss:%d' % (step,train_loss))
            
                x_pred = sess.run(x_conv,feed_dict={y:train_intensitys_batch,isTrain:False,keep_prob:1.0}) 
                temp = x_pred[0,:,:,:]
                temp = np.reshape(temp,[label_W,label_H])
                        
                x_real = train_labels_batch[0,:,:,:]
                x_real = np.reshape(x_real,[label_W,label_H])

                plt.subplot(121)
                plt.imshow(x_real)
                plt.subplot(122)
                plt.imshow(temp)
                plt.show()
            
            sess.run(train_op, feed_dict={y:train_intensitys_batch,x_label:train_labels_batch,isTrain:True,keep_prob:0.9}) 

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
        coord.request_stop()
    finally:
        coord.request_stop()
    coord.join(threads)

    saver.save(sess, model_save_path)


