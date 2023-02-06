import tensorflow as tf



def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def softmax_layer(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.zeros([shape[1]]))

    fc_h = tf.nn.softmax(tf.matmul(inpt, fc_w) + fc_b)

    return fc_h

def conv_layer(inpt, filter_shape, stride,isTrain=True):
    with tf.name_scope('conv_bn_relu'):
        filter_ = weight_variable(filter_shape)
        conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    
        batch_norm = tf.layers.batch_normalization(conv, training=isTrain, momentum=0.9)
    
        out = tf.nn.relu(batch_norm)
        return out

def residual_block(inpt, output_depth, down_sample, projection=False,isTrain=True):
    with tf.name_scope('residual_block'):
        input_depth = inpt.get_shape().as_list()[3]
        if down_sample:
            filter_ = [1,2,2,1]
            inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')
    
        conv1 = conv_layer(inpt, [5, 5, input_depth, output_depth], 1, isTrain)
        conv2 = conv_layer(conv1, [5, 5, output_depth, output_depth], 1, isTrain)
    
        if input_depth != output_depth:
            if projection:
                # Option B: Projection shortcut
                input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth], 2, isTrain)
            else:
                # Option A: Zero-padding
                input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
        else:
            input_layer = inpt
    
        res = conv2 + input_layer
        return res


# 定义偏置的函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    
# ResNet architectures used for CIFAR-10
def inference(inpt, batch_size, isTrain, keep_prob, length):
    
    inpt = tf.reshape(inpt, shape=[batch_size,length])
    
    with tf.variable_scope('FC1'):
        fc1 = tf.reshape(inpt, shape=[batch_size,length])
        W_fc1 = weight_variable([length,32*32])
        b_fc1 = bias_variable([32*32])
        
        fc1 = tf.matmul(fc1,W_fc1) + b_fc1
        bn1 = tf.layers.batch_normalization(fc1,training = isTrain)
        fc1 = tf.nn.relu(bn1)
        fc1 = tf.nn.dropout(fc1,keep_prob)
        
    with tf.variable_scope('FC2'):
        W_fc2 = weight_variable([32*32,64*64])
        b_fc2 = bias_variable([64*64])
        
        fc2 = tf.matmul(fc1,W_fc2) + b_fc2
        bn2 = tf.layers.batch_normalization(fc2,training = isTrain)
        fc2 = tf.nn.relu(bn2)
        fc2 = tf.nn.dropout(fc2,keep_prob)

    fc2 = tf.reshape(fc2, shape=[batch_size,64,64,1])
    
    with tf.variable_scope('conv1'):
        conv1 = conv_layer(fc2, [5, 5, 1, 16], 1, isTrain=isTrain)
        conv1 = tf.nn.dropout(conv1,keep_prob)
    
    with tf.variable_scope('conv2'):
        conv2 = conv_layer(conv1, [5, 5, 16, 32], 1, isTrain=isTrain)
        conv2 = tf.nn.dropout(conv2,keep_prob)
    
    #对数据进行分流处理    
    with tf.variable_scope('Max_Pooling'):
        M0 = conv2                                    #图像缩小1倍 64
        M1 = max_pool_2x2(M0)                         #图像缩小2倍 32 
        M2 = max_pool_2x2(M1)                         #图像缩小4倍 16
        M3 = max_pool_2x2(M2)                         #图像缩小8倍 8
        
    layers = []
    layers.append(M0)        
    with tf.variable_scope('Data_Flow_1'):
        for i in range (2):
            with tf.variable_scope('conv3_%d' % (i+1)):
                conv3_x = residual_block(layers[-1], 32, False,isTrain=isTrain)
                conv3 = residual_block(conv3_x, 32, False,isTrain=isTrain)
                layers.append(conv3_x)
                layers.append(conv3)
        conv3 = layers[-1]
    
    layers = []
    layers.append(M1)    
    with tf.variable_scope('Data_Flow_2'):
        for i in range (2):
            with tf.variable_scope('conv4_%d' % (i+1)):
                conv4_x = residual_block(layers[-1], 32, False,isTrain=isTrain)
                conv4 = residual_block(conv4_x, 32, False,isTrain=isTrain)
                layers.append(conv4_x)
                layers.append(conv4)
                
        with tf.variable_scope('Upsampling_1'):
            W_de_conv4 = weight_variable([3, 3, 32, 32])
            h_de_conv4 = tf.nn.conv2d_transpose(layers[-1],W_de_conv4,output_shape=[batch_size, 64, 64, 32],strides=[1,2,2,1],padding="SAME")
            h_de_conv4 = tf.nn.relu(h_de_conv4)           
        conv4 = h_de_conv4   
        
    layers = []
    layers.append(M2)    
    with tf.variable_scope('Data_Flow_3'):
        for i in range (2):
            with tf.variable_scope('conv5_%d' % (i+1)):
                conv5_x = residual_block(layers[-1], 32, False,isTrain=isTrain)
                conv5 = residual_block(conv5_x, 32, False,isTrain=isTrain)
                layers.append(conv5_x)
                layers.append(conv5)
                
        with tf.variable_scope('Upsampling_2'):
            W_de_conv5 = weight_variable([3, 3, 32, 32])
            h_de_conv5 = tf.nn.conv2d_transpose(layers[-1],W_de_conv5,output_shape=[batch_size, 32, 32, 32],strides=[1,2,2,1],padding="SAME")
            h_de_conv5 = tf.nn.relu(h_de_conv5)
            
        with tf.variable_scope('Upsampling_3'):
            W_de_conv5_1 = weight_variable([3, 3, 32, 32])
            h_de_conv5_1 = tf.nn.conv2d_transpose(h_de_conv5,W_de_conv5_1,output_shape=[batch_size, 64, 64, 32],strides=[1,2,2,1],padding="SAME")
            h_de_conv5_1 = tf.nn.relu(h_de_conv5_1)               
        conv5 = h_de_conv5_1             
                
    layers = []
    layers.append(M3)    
    with tf.variable_scope('Data_Flow_4'):
        for i in range (2):
            with tf.variable_scope('conv6_%d' % (i+1)):
                conv6_x = residual_block(layers[-1], 32, False,isTrain=isTrain)
                conv6 = residual_block(conv6_x, 32, False,isTrain=isTrain)
                layers.append(conv6_x)
                layers.append(conv6)
                
        with tf.variable_scope('Upsampling_4'):
            W_de_conv6 = weight_variable([3, 3, 32, 32])
            h_de_conv6 = tf.nn.conv2d_transpose(layers[-1],W_de_conv6,output_shape=[batch_size, 16, 16, 32],strides=[1,2,2,1],padding="SAME")
            h_de_conv6 = tf.nn.relu(h_de_conv6)
            
        with tf.variable_scope('Upsampling_5'):
            W_de_conv6_1 = weight_variable([3, 3, 32, 32])
            h_de_conv6_1 = tf.nn.conv2d_transpose(h_de_conv6,W_de_conv6_1,output_shape=[batch_size, 32, 32, 32],strides=[1,2,2,1],padding="SAME")
            h_de_conv6_1 = tf.nn.relu(h_de_conv6_1)   

        with tf.variable_scope('Upsampling_6'):
            W_de_conv6_2 = weight_variable([3, 3, 32, 32])
            h_de_conv6_2 = tf.nn.conv2d_transpose(h_de_conv6_1,W_de_conv6_2,output_shape=[batch_size, 64, 64, 32],strides=[1,2,2,1],padding="SAME")
            h_de_conv6_2 = tf.nn.relu(h_de_conv6_2)             
        conv6 = h_de_conv6_2
    
    #对网络进行链接，之后卷积操作减少通道数  
    Res = tf.concat([conv3,conv4,conv5,conv6],3) 
    
    with tf.variable_scope('conv7'):
        conv7 = conv_layer(Res, [5, 5, 128, 64], 1, isTrain=isTrain)
        conv7 = tf.nn.dropout(conv7,keep_prob)
        
    with tf.variable_scope('Max_Pooling_1'):
        Maxpool_1 = max_pool_2x2(conv7)                                   
    
    with tf.variable_scope('conv8'):
        conv8 = conv_layer(Maxpool_1, [5, 5, 64, 64], 1, isTrain=isTrain)
        conv8 = tf.nn.dropout(conv8,keep_prob)
            
    with tf.variable_scope('Max_Pooling_2'):
        Maxpool_2 = max_pool_2x2(conv8)
 
    with tf.variable_scope('Upsampling_7'):
        W_de_conv7 = weight_variable([3, 3, 1, 64])
        h_de_conv7 = tf.nn.conv2d_transpose(Maxpool_2,W_de_conv7,output_shape=[batch_size, 32, 32, 1],strides=[1,2,2,1],padding="SAME")
        h_de_conv7 = tf.nn.relu(h_de_conv7)
        h_de_conv7 = tf.nn.dropout(h_de_conv7, keep_prob)
                
        assert h_de_conv7.get_shape().as_list()[1:] == [32, 32, 1]
      
    return h_de_conv7
