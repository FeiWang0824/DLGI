
# coding: utf-8

# In[1]:
import os
import tensorflow as tf
from PIL import Image  #注意Image,后面会用到

# 生成字符串型的属性
def _bytes_feature(value):
     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# 生成整型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def generate_TFRecordfile(image_path,label_path,TFRecord_path):
    images = []
    labels = []
    
    # 获取每一个样本的路径
    for file in os.listdir(image_path):
        images.append(image_path+file)
    for file in os.listdir(label_path):
        labels.append(label_path+file)
    num_examples = len(images)
    
    print('There are %d files\n'%(num_examples))
    
    writer = tf.python_io.TFRecordWriter(TFRecord_path)#创建一个writer写TFRecord文件
    for index in range(num_examples):  
        image = Image.open(images[index])
        image = image.tobytes()                
        label = Image.open(labels[index])
        label = label.tobytes()
        
        #将一个样例转换为Example Protocol Buffer的格式，并且将所有信息写入这个数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            'image':_bytes_feature(image),
            'label':_bytes_feature(label)}))
        
        writer.write(example.SerializeToString())#将一个Example 写入TFRecord文件
#     print('TFRecord file was generated successfully\n')
    writer.close()

def get_batch(TFRecord_path, img_W,img_H, label_W,label_H, batch_size,num_epochs):
    
    reader = tf.TFRecordReader() # 创建一个reader来读取TFRecord文件中的样例 
    
    files = tf.train.match_filenames_once(TFRecord_path) # 获取文件列表
    filename_queue = tf.train.string_input_producer(files,shuffle = False,num_epochs = num_epochs) # 创建文件名队列，乱序，每个样本使用num_epochs次
    
    # 读取并解析一个样本
    _,example = reader.read(filename_queue)
    features = tf.parse_single_example(
        example,
        features={
            'image':tf.FixedLenFeature([],tf.string),
            'label':tf.FixedLenFeature([],tf.string)})
    
    # 使用tf.decode_raw将字符串解析成图像对应的像素数组 （）
    images = tf.decode_raw(features['image'],tf.int32) 
    labels = tf.decode_raw(features['label'],tf.uint8)
    

    #所得像素数组为shape为(（img_W*img_H）,)，应该reshape
    images = tf.reshape(images, shape=[img_W,img_H])
    labels = tf.reshape(labels, shape=[label_W,label_H])
    
#    在这里添加图像预处理函数（optional）
    # 归一化
    #images = images/tf.reduce_max(images)  # CNN_3232_64_normal.ckpt
    
    images = tf.to_float(images)
    images_std = tf.sqrt(tf.reduce_sum(tf.square(images - tf.reduce_mean(images)))/64)        # CNN_3232_64_bzh.ckpt
    images = (images-tf.reduce_mean(images))/images_std                                       # CNN_3232_64_bzh.ckpt
    
#    print(np.max(images)-np.min(images))
#
#          
#    使用 tf.train.batch函数来组合样例
#    这里不使用 tf.train.shuffle_batch是因为通过tf.train.string_input_producer创建文件名队列时应将乱序过了。
#    但是实验发现每次运行程序都是得到相同的mini-batch，因此考虑使用tf.train.shuffle_batch
    Image_Batch,Label_Batch = tf.train.batch([images,labels],
                                             batch_size = batch_size,
                                             num_threads = 5,
                                             capacity = 100+3*batch_size)
    
    
    return Image_Batch,Label_Batch

