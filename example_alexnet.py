import tensorflow as tf 

def inference(images):
    '''
    Alexnet模型
    输入：images的tensor
    返回：Alexnet的最后一层卷积层
    '''
    parameters=[]

    #conv1
    with tf.name_scope('conv1') as scope:
        kernel=tf.Variable(tf.truncated_normal([11,11,3,64],dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(images,kernel,[1,4,4,1],padding='same')
        biases=tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv1=tf.nn.relu(bias,name=scope)
        print_activations(conv1)
        parameters+=[kernel,biases]
    #LRN1
    with tf.name_scope('LRN1') as scope:
        LRN1=tf.nn.local_response_normalization(conv1,alpha=1e-4,beta=0.75,depth_radius=2,bias=2.0)
    #pool1
    pool1=tf.nn.max_pool(LRN1,ksize[1,3,3,1],strides=[1,2,2,1],padding='valid',name='pool1')
    print_activations(pool1)

    #conv2
    with tf.name_scope('conv2') as scope:
        kernel=tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='same')
        biases=tf.Variable(tf.constant(0.0,shape=[192],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv2=tf.nn.relu(bias,name=scope)
        parameters+=[kernel,biases]
        print_activations(conv2)
    #LRN2
    with tf.name_scope('LRN2') as scope:
        LRN2=tf.nn.local_response_normalization(conv2,alpha=1e-4,beta=0.75,depth_radius=2,bias=2.0)
    #pool2
    pool2=tf.nn.max_pool(LRN2,ksize[1,3,3,1],strides=[1,2,2,1],padding='valid',name='pool2')
    print_activations(pool2) 


    #conv3
    with tf.name_scope('conv3') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,192,384],dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='same')
        biases=tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv2=tf.nn.relu(bias,name=scope)
        parameters+=[kernel,biases]
        print_activations(conv3)

    #conv4
    with tf.name_scope('conv4') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,384,256],dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='same')
        biases=tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv2=tf.nn.relu(bias,name=scope)
        parameters+=[kernel,biases]
        print_activations(conv4)
 
    #conv5
    with tf.name_scope('conv5') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,256,256],dtype=tf.float32,stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding='same')
        biases=tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv2=tf.nn.relu(bias,name=scope)
        parameters+=[kernel,biases]
        print_activations(conv5)
    #pool5
    pool5=tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='valid',name='pool5')
    print_activations(pool5)

    return pool5,parameters

    