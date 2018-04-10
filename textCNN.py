import tensorflow as tf 
import numpy as np 

class textCNN(object):
    def __init__(self,sequence_length,num_classes,vocab_size,embedding_size,filter_sizes,num_filters):
        self.input_x=tf.placeholder(tf.int32,[None,sequence_length],name="input_x")
        self.input_y=tf.placeholder(tf.float32,[None,num_classes],name="input_y")
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")

        with tf.device('/cpu:0'),tf.name_scope("embedding"):
            W=tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),name="W")
            # 随机均匀分布初始化
            self.embedded_chars=tf.nn.embedding_lookup(W,self.input_x)
            # 创建了实际的嵌入操作，输出结果为3D张量，多了channel维，则[None,sequence_length,embedding_size,1]
            self.embedded_chars_expanded=tf.expand_dims(self.embedded_chars,-1)

# 每一个卷积核有多种不同的尺寸，每个卷积产生的张量形状不一，需要迭代对每一个创建一层，再融合到一个大特征向量

        pooled_outputs=[]
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s"%filter_size):
                filter_shape=[filter_size,embedding_size,1,num_filters]
                W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="W")
                b=tf.Variable(tf.constant(0.1,shape=[num_filters]),name="b")
                conv=tf.nn.conv2d(self.embedded_chars_expanded,W,strides=[1,1,1,1],padding="VALID",name="conv")
                # 不在边缘做填补，窄卷积，输出形状[1,sequence_length-filter_size+1,1,1]
                h=tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")
                pooled=tf.nn.max_pool(h,ksize=[1,sequence_length-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID',name="pool")
                # maxpool之后变成了[batch_size,1,1,num_filters],最后一位对应特征
                pooled_outputs.append(pooled)
        
        num_filters_total=num_filters*len(filter_sizes)
        self.h_pool=tf.concat(3,pooled_outputs)
        self.h_pool_flat=tf.reshape(self.h_pool,[-1,num_filters_total])
        # 将tensor组合成一个很长的特征，形如[batch_size,num_filter_total],将高维向量铺平，tf.reshape设置为-1

        with tf.name_scope("dropout"):
            self.h_drop=tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)

# 矩阵乘法选择分数最高的类，做个预测；也可以用softmax把数据转化为正规化的概率，但这并不会改变最终的预测结果。

        with tf.name_scope("output"):
            W=tf.Variable(tf.truncated_normal([num_filters_total,num_classes],stddev=0.1),name="W")
            b=tf.Variable(tf.constant(0.1,shape=[num_classes]),name="b")
            self.scores=tf.nn.xw_plus_b(self.h_drop,W,b,name="scores")
            self.predictions=tf.argmax(self.scores,1,naem="predictions")
        
        with tf.name_scope("loss"):
            losses=tf.nn.softmax_cross_entropy_with_logits(self.scores,self.input_y)
            self.loss=tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_predictions=tf.equal(self.predictions,tf.argmax(self.input_y,1))
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions,"float"),name="accuracy")
        
        with tf.Graph().as_default():
            session_conf=tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                # 允许TensorFlow在指定设备不存在时自动调整设备。
                log_device_placement=FLAGS.log_device_placement
                # 会在指定设备上存储日志文件，对debug很有帮助。FLAGS是程序接收的命令行参数
            )
            sess=tf.Session(config=session_conf)
            with sess.as_default():

cnn=textCNN(
    sequence_length=x_train.shape[1],
    num_classes=2,
    vocab_size=len(vocabulary),
    embedding_size=FLAGS.embedding_dim,
    filter_sizes=map(int,FLAGS.filter_sizes.split(",")),
    num_filters=FLAGS.num_filters
)

global_step=tf.Variable(0,name="global_step",trainable=False)
optimizer=tf.train.AdamOptimizer(1e-4)
grads_and_vars=optimizer.compute_gradients(cnn.loss)
train_op=optimizer.apply_gradients(grads_and_vars,global_step=global_step)
# 每一次运行train_op就是一次训练，tf自动识别哪些参数是可训练的，然后计算梯度。global_step是用来计数

timestamp=str(int(time.time()))
out_dir=os.path.abspath(os.path.join(os.path.curdir,"runs",timestamp))
print("Writing to {}\n".format(out_dir))

loss_summary=tf.scalar_summary("loss",cnn.loss)
acc_summary=tf.scalar_summary("accuracy",cnn.accuracy)

train_summary_op=tf.merge_summary([loss_summary,acc_summary])
train_summary_dir=os.path.join(out_dir,"summaries","train")
train_summary_writer=tf.train.SummaryWiter(train_summary_dir,sess.grah_def)

dev_summary_op=tf.merge_summary([loss_summary,acc_summary])
dev_summary_dir=os.path.join(out_dir,"summaries","dev")
dev_summary_writer=os.train.SummaryWiter(dev_summary_dir,sess.graph_def)

checkpoint_dir=os.path.abspath(os.path.join(out_dir,"checkpoints"))
checkpoint_prefix=os.path.join(checkpoint_dir,"model")

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver=tf.train.Saver(tf.all_variables())

sess.run(tf.initialize_all_tables())

def train_step(x_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, summaries, loss, accuracy = sess.run(
        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    train_summary_writer.add_summary(summaries, step)

def dev_step(x_batch, y_batch, writer=None):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropout_keep_prob: 1.0
    }
    step, summaries, loss, accuracy = sess.run(
        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    if writer:
        writer.add_summary(summaries, step)

atch_iter(
    zip(x_train, y_train), FLAGS.batch_size, FLAGS.num_epochs)
# Training loop. For each batch...
for batch in batches:
    x_batch, y_batch = zip(*batch)
    train_step(x_batch, y_batch)
    current_step = tf.train.global_step(sess, global_step)
    if current_step % FLAGS.evaluate_every == 0:
        print("\nEvaluation:")
        dev_step(x_dev, y_dev, writer=dev_summary_writer)
        print("")
    if current_step % FLAGS.checkpoint_every == 0:
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))       