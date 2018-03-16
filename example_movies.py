import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import tensorflow as tf 
import os 
import pickle
import re 
from tensorflow.python.ops import math_ops
from urllib.request import urlretrieve
from os.path import isfile,isdir 
from tqdm import tqdm
import zipfile
import hashlib
'''
def _unzip(save_path,_,database_name,data_path):
    print('Extracting {} ...'.format(database_name))
    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(data_path)

def download_extract(database_name,data_path):
    dataset_ml='m1-1m'
    if database_name==dataset_ml:
        url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        hash_code = 'c4d9eecfca2ab87c1945afe126590906'
        extract_path = os.path.join(data_path, 'ml-1m')
        save_path = os.path.join(data_path, 'ml-1m.zip')
        extract_fn = _unzip

    if os.path.exists(extract_path):
        print('Found {} Data'.format(database_name))
        return 
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(save_path):
        with DLProgress(unit='B',unit_scale=True,miniters=1,desc='Downloading {}'.format(database_name)) as pbar:
            urlretrieve(
                url,
                save_path,
                pbar.hook
            )
    
    assert hashlib.md5(open(save_path,'rb').read()).hexdigest()==hash_code,\
        '{} file is corrupted. Remove the file and try again'.format(save_path)
    
    os.makedirs(extract_path)
    try:
        extract_fn(save_path,extract_path,database_name,data_path)
    except Exception as err:
        shutil.rmtree(extract_path)
        raise err
    
    print('Done.')

class DLProgress(tqdm):
    last_block=0
    def hook(self,block_num=1,block_size=1,total_size=None):
        self.total=total_size
        self.update((block_num-self.last_block)*block_size)
        self.last_block=block_num

data_dir='./'
download_extract('m1-1m',data_dir)
'''

'''
users_title = ['UserID', 'Gender', 'Age', 'OccupationID', 'Zip-code']
users = pd.read_table('E:/dataset/NLP/movies/ml-1m/users.dat', sep='::', header=None, names=users_title, engine = 'python')
print(users.head(5))

movies_title = ['MovieID', 'Title', 'Genres']
movies = pd.read_table('E:/dataset/NLP/movies/ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine = 'python')
print(movies.head(5))

ratings_title = ['UserID','MovieID', 'Rating', 'timestamps']
ratings = pd.read_table('E:/dataset/NLP/movies/ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine = 'python')
print(ratings.head(5))
'''

def load_data():

    # read users data
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_table('E:/dataset/NLP/movies/ml-1musers.dat', sep='::', header=None, names=users_title, engine = 'python')
    users = users.filter(regex='UserID|Gender|Age|JobID')
    users_orig = users.values

    #handle the data of gender and age
    gender_map = {'F':0, 'M':1}
    users['Gender'] = users['Gender'].map(gender_map)
    age_map = {val:ii for ii,val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age_map)

    #read movies data
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_table('E:/dataset/NLP/movies/ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine = 'python')
    movies_orig = movies.values

    #delete the years in title
    pattern = re.compile(r'^(.*)\((\d+)\)$')
    title_map = {val:pattern.match(val).group(1) for ii,val in enumerate(set(movies['Title']))}
    movies['Title'] = movies['Title'].map(title_map)

    #inverse the genres's data into its dictionary 
    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)

    genres_set.add('<PAD>')
    genres2int = {val:ii for ii, val in enumerate(genres_set)}

    #inverse the genres into the same length list 18 
    # (a movie belongs to some genres)
    genres_map = {val:[genres2int[row] for row in val.split('|')] for ii,val in enumerate(set(movies['Genres']))}

    for key in genres_map:
        for cnt in range(max(genres2int.values()) - len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key]) + cnt,genres2int['<PAD>'])
    
    movies['Genres'] = movies['Genres'].map(genres_map)

    #create the titles' dictionary
    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)
    
    title_set.add('<PAD>')
    title2int = {val:ii for ii, val in enumerate(title_set)}

    #inverse the titles into the same length list(15)
    title_count = 15
    title_map = {val:[title2int[row] for row in val.split()] for ii,val in enumerate(set(movies['Title']))}
    
    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt,title2int['<PAD>'])
    
    movies['Title'] = movies['Title'].map(title_map)

    #read the ratings
    ratings_title = ['UserID','MovieID', 'ratings', 'timestamps']
    ratings = pd.read_table('./ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine = 'python')
    ratings = ratings.filter(regex='UserID|MovieID|ratings')

    #merge the three tables
    data = pd.merge(pd.merge(ratings, users), movies)
    
    #plit the data to X,y
    target_fields = ['ratings']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]
    
    features = features_pd.values
    targets_values = targets_pd.values
    
    return title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig

title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = load_data()
pickle.dump((title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig), open('E:/dataset/NLP/movies/ml-1m/preprocess.p', 'wb'))

print(users.head(5))
print(movies.head5))
print(movies.values[0])

title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(open('E:/dataset/NLP/movies/ml-1m/preprocess.p', mode='rb'))



def save_params(params):
    pickle.dump(params,open('params.p','wb'))

def load_params():
    return pickle.load(open('params.p',mode='rb'))

# userID，movieID等转成one hot编码会非常稀疏，
# 则将这些字段转成数字，用这个数字当作嵌入矩阵的索引。
# 文本卷积
print("======features=========")
print(features.shape)

#embedding dim 
emded_dim=32
userid_max=max(features.take(0,1))+1
gender_max=max(features.take(2,1))+1
age_max=max(features.take(3,1))+1
job_max=max(features.take(4,1))+1

movieid_max=max(features.take(1,1))
movie_categories_max=max(genres2int.values())+1
movie_title_max=len(title_set)
#handle the many genres, sum    mean??
combiner="sum"
sentences_size=title_count

# text filters
window_sizes={2,3,4,5}
filter_num=8

movieid2idx={val[0]:i for i,val in enumerate(movies.values)}

num_epochs=5
batch_size=256
dropout_keep=0.5
learning_rate=0.0001
show_every_n_batches=20

save_dir='E:/ML/models/nlp/movies'

def get_inputs():
    user_id=tf.placeholder(tf.int32,[None,1],name="user_id")
    user_gender=tf.placeholder(tf.int32,[None,1],name="user_gender")
    user_age=tf.placeholder(tf.int32,[None,1],name="user_age")
    user_job=tf.placeholder(tf.int32,[None,1],name="user_job")

    movie_id=tf.placeholder(tf.int32,[None,1],name="movie_id")
    movie_categories=tf.placeholder(tf.int32,[None,18],name="move_categories")
    movie_titles=tf.placeholder(tf.int32,[None,15],name="movie_title")
    targets=tf.placeholder(tf.int,[None,1],name="targets")
    lr=tf.placeholder(tf.float32,name="learningrate")
    dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
    return user_id,user_gender,user_age,user_job,movie_id,movie_categories,movie_title,targets,lr,dropout_keep_prob

def get_user_embedding(user_id,user_gender,user_age,user_job):
    with tf.name_scope("user_embedding"):
        user_id_embed_matrix=tf.Variable(tf.random_uniform([userid_max,emded_dim],-1,1),name="user_id_embed_matrix")
        user_id_embed_layer=tf.embedding_lookup(user_id_embed_matrix,user_id,name="user_id_embed_layer")

        gender_embed_matrix=tf.Variable(tf.random_uniform([gender_max,embed_dim//2],-1,1),name="gender_embed_matrix")
        gender_embed_layer=tf.nn.embedding_lookup(gender_embed_matrix,user_gender,name="gender_embed_layer")

        age_embed_matrix=tf.Variable(tf.random_uniform([age_max,embed_dim//2],-1,1),name="age_embed_matrix")
        age_embed_layer=tf.nn.embedding_lookup(age_embed_matrix,user_age,name="age_embed_layer")

        job_embed_matrix=tf.Variable(tf.random_uniform([job_max,embed_dim//2],-1,1),name="job_embed_matrix")
        job_embed_layer=tf.nn.embedding_lookup(job,user_job,name="job_embed_layer")

    return user_id_embed_layer,gender_embed_layer,age_embed_layer,job_embed_layer

def get_user_feature_layer(user_id_embed_layer,gender_embed_layer,age_embed_layer,job_embed_layer):
    with tf.name_scope("user_fc"):

        # the first full-connection layer
        userid_fc_layer=tf.layers.dense(user_id_embed_layer,embed_dim,name="userid_fc_layer",activation=tf.nn.relu)
        gender_fc_layer=tf.layers.dense(gender_embed_layer,embed_dim,name="gender_fc_layer",activation=tf.nn.relu)
        age_fc_layer=tf.layers.dense(age_embed_layer,embed_dim,name="age_fc_layer",activation=td.nn.relu)
        job_fc_layer=tf.layers.dense(job_embed_layer,embed_dim,name="job_fc_layer",activation=tf.nn.relu)
        
        # the second fc layer
        user_combine_layer=tf.concat([userid_fc_layer,gender_fc_layer,age_fc_layer,job_fc_layer],2)
        user_combine_layer=tf.contrib.layers.fully_connected(user_combine_layer,200,tf.tanh)

        user_combine_layer_flat=tf.reshape(user_combine_layer,[-1,200])
    return user_combine_layer,user_combine_layer_flat

def get_movie_id_embed_layer(movie_id):
    with tf.name_scope("movie_embedding"):
        movie_id_embed_matrix=tf.Variable(tf.random_uniform([movie_id_max,embed_dim],-1,1),name="movie_id_embed_matrix")
        movie_id_embed_layer=tf.nn.embedding_lookup(movie_id_embed_matrix,movie_id,name="movie_id_embed_layer")
    return movie_id_embed_layer

def get_movie_categories_layers(movie_categories):
    with tf.name_scope("movie_categories_layers"):
        movie_categories_embed_matrix=tf.Variable(tf.random_uniform([movie_categories_max,embed_dim],-1,1),name="movie_categories_embed_matrix")
        movie_categories_embed_layer=tf.nn.embedding_lookup(movie_categories_embed_matrix,movie_categories,name="movie_categories_embed_layer")
        if combiner=="sum":
            movie_categories_embed_layer=tf.reduce_sum(movie_categories_embed_layer,axis=1,keep_dims=True)
        # elif combiner=="mean":
    return movie_categories_embed_layer

def get_movie_cnn_layer(movies_title):
    with tf.name_scope("movie_embedding"):
        movie_title_embed_matrix=tf.Variable(tf.random_uniform([movie_title_max,embed_dim],-1,1),name="movie_title_embed_matrix")
        movie_title_embed_layer=tf.nn.embedding_lookup(movie_title_embed_matrix,movie_title,name="movie_titile_embed_layer")
        movie_title_embed_layer_expand=tf.expand_dims(movie_title_embed_layer,-1)

    # convolution and maxpooling
    pool_layer_lst=[]
    for window_size in window_sizes:
        with tf.name_scope("movie_txt_conv_maxpool_{}".format(window_size)):
            filter_weights=tf.Variable(tf.truncated_normal([window_size,embed_dim,1,filter_num],stddec=0.1),name="filter_weights")
            filter_bias=tf.Variable(tf.constant(0.1,shape=[filter_num]),name="filter_bias")

            conv_layer=tf.nn.conv2d(movie_title_embed_layer_expand,filter_weights,[1,1,1,1],padding="VALID",name="conv_layer")
            relu_layer=tf.nn.relu(tf.nn.bias_add(conv_layer,filter_bias),name="relu_layer")

            maxpool_layer=tf.nn.max_pool(relu_layer,[1,sentences_size-window_size+1,1,1],[1,1,1,1],padding="VALID",name="maxpool_layer")
            pool_layer_lst.append(maxpool_layer)

    # dropout
    with tf.name_scope("pool_dropout"):
        pool_layer=tf.concat(pool_layer_lst,3,name="pool_layer")
        max_num=len(window_sizes)*filter_num
        pool_layer_flat=tf.reshape(pool_layer,[-1,1,max_num],name="pool_layer_flat")
        
        dropout_layer=tf.nn.dropout(pool_layer_flat,dropout_keep_prob,name="dropout_layer")
    return pool_layer_flat,dropout_layer

def get_movie_feature_layer(movie_id_embed_layer,movie_categories_embed_layer,dropout_layer):
    with tf.name_scope("movie_fc"):
        # the first fc layer
        movie_id_fc_layer=tf.layer.dense(movie_id_embed_layer,embed_dim,name="movie_id_fc_layer",activation=tf.nn.relu)
        movie_categories_fc_layer=tf.layers.dense(movie_categories_embed_layer,embed_dim,name="movie_categories_fc_layer",activation=tf.nn.relu) 
    
        movie_combine_layer=tf.concat([movie_id_fc_layer,movie_categories_fc_layer,dropout_layer],2)
        movie_combine_layer=tf.contrib.layers.fully_connected(movie_combine_layer,200,tf.tanh)

        movie_combine_layer_flat=tf.reshape(movie_combine_layer,[-1,200])
    return movie_combine_layer,movie_combine_layer_flat

tf.reset_default_graph()
train_graph=tf.Graph()
with train_graph.as_default():
    # placeholder
    user_id,user_gender,user_age,user_job,movie_id,movie_categories,movies_title,targets,lr,dropout_keep_prob=get_inputs()
    # get four users' embedding variabels
    user_id_embed_layer.gender_embed_layer,age_embed_layer,job_embed_layer=get_user_embedding(user_id,user_gender,user_age,user_job)
    # get the users'features
    user_combine_layer,user_combine_layer_flat=get_user_feature_layer(user_id_embed_layer,gender_embed_layer,age_embed_layer,job_embed_layer)
    # get the movie_id's embedding vector
    movie_id_embed_layer=get_movie_id_embed_layer(movie_id)
    # get the movies'titles feature vector
    pool_layer_flat,dropout_layer=get_movie_cnn_layer(movies_title)
    # get the movies'feature
    movie_combine_layer,movie_combine_layer_flat=get_movie_feature_layer(movie_id_embed_layer,movie_categories_embed_layer,dropout_layer)

    with tf.name_scope("inference"):
        # the first method


        # the second method
        inference=tf.reduce_sum(user_combine_layer_flat*movie_combine_layer_flat,axis=1)
        inference=tf._expand_dims(inference,axis=1)

    with tf.name_scope("loss"):
        cost=tf.losses.mean_squared_error(targets,inference)
        loss=tf.reduce_mean(cost)
        # train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    
    global_step=tf.Variable(0,name="global_step",trainable=False)
    optimizer=tf.train.AdamOptimizer(lr)
    gradients=optimizer.compute_gradients(loss)
    train_op=optimizer.apply_gradients(gradients,global_step=global_step)

def get_batches(Xs,ys,batch_size):
    for start in range(0,len(Xs),batch_size):
        end=min(start+batch_size,len(Xs))
        yeild Xs[start:end],ys[start:end]



import matplotlib.pyplot as plt
import time
import datetime

losses={'train':[],'test':[]}

with tf.Session(graph=train_graph) as sess:
    # for tensorboard
    # keep track of grandient values and sparsity
    grad_summaries=[]
    for g,v in gradients:
        grad_hist_summary=tf.summary.histogram("{}/grad/hist".format(v.name.replace(':','-')),g)
        sparsity_summary=tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(':','-')),tf.nn.zero_fraction(g))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
    grad_summaries_merged=tf.summary.merge(grad_summaries)

    # output directory for models and summaries
    timestamp=str(int(time.time()))
    out_dir=os.path.abspath(os.path.join(os.path.curdir,"runs",timestamp))
    print("writting to {}\n".format(out_dir))
    
    # summaries for loss and accuracy
    loss_summary=tf.summary.scalar("loss",loss)

    # train summaries
    train_summary_op=tf.summary.merge([loss_summary,grad_summaries_merged])
    train_summary_dir=os.path.join(out_dir,"summaries","train")
    train_summary_writer=tf.summary.FileWriter(train_summary_dir,sess.graph)

    # inference summaries
    inference_summary_op=tf.summary.merge([loss_summary])
    inference_summary_dir=os.path.join(out_dir,"summaries","inference")
    inference_summary_writer=tf.summary.FileWriter(inference_summary_dir,sess.graph)

    sess.run(tf.global_variables_initializer())
    saver=tf.train.Saver()
    for epoch_i in range(num_epochs):
        train_X,test_X,train_y,test_y=train_test_split(features,targets_values,test_size=0.2,random_state=0)
        train_batches=get_batches(train_X,train_y,batch_size)
        test_batches=get_batches(test_X,test_y,batch_size)

        for batch_i in range(len(train_X)//batch_size):
            x,y=next(train_batches)
            
            categories=np.zeros([batch_size,18])
            for i in range(batch_size):
                categories[i]=x.take(6,1)[i]

                titles=np.zeros([batch_size,18])
                for i in range(batch_size):
                    titles[i]=x.take(5,1)[i]

                feed={
                    user_id:np.reshape(x.take(0,1),[batch_size,1]),
                    user_gender:np.reshape(x.take(2,1),[batch_size,1]),
                    user_age:np.reshape(x.take(3,1),[batch_size,1]),
                    user_job:np.reshape(x,take(4,1),[batch_size,1]),
                    movie_id:np.reshape(x,take(1,1),[batch_size,1]),
                    movie_categories:categories,
                    movies_title:titles,
                    targets:np.reshape(y,[batch_size,1]),
                    dropout_keep_prob:dropout_keep,
                    lr:lr
                }
            step,train_loss,summaries,_=sess.run([global_step,loss,train_summary_op,train_op],feed)
            losses['train'].append(train_loss)
            train_summary_writer.add_summary(summaries,step)

            if(epoch_i*(len(train_X)//batch_size)+batch_i)%show_every_n_batches==0:
                time_str=datetime.datetime.now().isoformat()
                print('{}:epoch {:>3} batch {:>4}/{}  train_loss={:.3f}'.format(
                    time_str,
                    epoch_i,
                    batch_i,
                    (len(train_X)//batch_size),
                    train_loss
                ))
        
        for batch_i in range(len(test_X)//batch_size):
            x,y=next(test_batches)

            categories=np.zeros([batch_size,18])
            for i in range(batch_size):
                categories[i]=x.take(6,1)[i]
            
            titles=np.zeros([batch_size,sentences_size])
            for i in range(batch_size):
                titles[i]=x.take(5,1)[i]
            
            feed={
                user_id:np.reshape(x.take(0,1),[batch_size,1]),
                user_gender:np.reshape(x.take(2,1),[batch_size,1]),
                user_age:np.reshape(x.take(3,1),[batch_size,1]),
                user_job:np.reshape(x,take(4,1),[batch_size,1]),
                movie_id:np.reshape(x,take(1,1),[batch_size,1]),
                movie_categories:categories,
                movies_title:titles,
                targets:np.reshape(y,[batch_size,1]),
                dropout_keep_prob:dropout_keep,
                lr:lr
            }
        
        step,test_loss,summaries=sess.run([global_step,loss,inference_summary_op],feed)
        lossed['test'].append(test_loss)
        inference_summary_writer.add_summary(summaries,step)

        time_str=datetime.datetime.now().isoformat()
        if(epoch_i*(len(test_X)//batch_size)+batch_i)%show_every_n_batches==0:
            time_str=datetime.datetime.now().isoformat()
            print('{}:epoch {:>3} batch {:>4}/{}  test_loss={:.3f}'.format(
                time_str,
                epoch_i,
                batch_i,
                (len(train_X)//batch_size),
                test_loss
            ))   

    saver.save(sess,save_dir)
    print('Model Trained and Saved')    


save_params((save_dir))
load_dir=load_params()

plt.plot(losses['train'],label='training loss')
plt.legend()
_=plt.ylim()

plt.plot(losses['test'],label='test loss')
plt.legend()
_=plt.ylim()

def get_tensors(loaded_graph):
    user_id=loaded_graph.get_tensors_by_name("user_id:0")
    user_gender=loaded_graph.get_tensors_by_name("user_gender:0")
    user_age=loaded_graph.get_tensors_by_name("user_age:0")
    user_job=loaded_graph.get_tensors_by_name("user_job:0")
    movie_id=loaded_graph.get_tensors_by_name("movie_id:0")
    movie_categories=loaded_graph.get_tensors_by_name("movie_categories:0")
    movies_title=loaded_graph.get_tensors_by_name("movies_title:0")
    targets=loaded_graph.get_tensors_by_name("targets:0")
    dropout_keep_prob=loaded_graph.get_tensors_by_name("dropout_keep_prob:0")
    lr=loaded_graph.get_tensors_by_name("lr:0")

    # inference=loaded_graph.get_tensors_by_name("inference/inference/BiasAdd:0")
    inference=loaded_graph.get_tensors_by_name("inference/ExpandDims:0")
    movie_combine_layer_flat=loaded_graph.get_tensors_by_name("movie_fc/reshape:0")
    user_combine_layer_flat=loaded_graph.get_tensors_by_name("user_fc/reshape:0")

    return user_id,user_gender,user_age,user_job,movie_id,movie_categories,movies_title,targets,lr,dropout_keep_prob,inference,movie_combine_layer_flat,user_combine_layer_flat

# forwards
def rating_movie(user_id_val,movie_id_val):
    loaded_graph=tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        loader=tf.train.import_meta_graph(load_dir+'.meta')
        loader.restore(sess,load_dir)

        user_id,user_gender,user_age,user_job,movie_id,movie_categories,movies_title,targets,lr,dropout_keep_prob,inference,_,__=get_tensors(loaded_graph)

        categories=np.zeros([1,18])
        categories[0]=movies.values[movieid2idx[movie_id_val]][1]

        titles=np.zeros([1,18])
        titles[0]=movies.values[movieid2idx[movie_id_val]][1]

        feed={
            user_id: np.reshape(users.values[user_id_val-1][0], [1, 1]),
            user_gender: np.reshape(users.values[user_id_val-1][1], [1, 1]),
            user_age: np.reshape(users.values[user_id_val-1][2], [1, 1]),
            user_job: np.reshape(users.values[user_id_val-1][3], [1, 1]),
            movie_id: np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
            movie_categories: categories,  #x.take(6,1)
            movies_title: titles,  #x.take(5,1)
            dropout_keep_prob: 1}
        }
        # get prediction
        inference_val=sess.run([inference],feed)

        return (inference_val)

loaded_graph=tf.Graph()
movie_matrix=[]
with tf.Session(graph=loaded_graph) as sess:
    loader=tf.train.import_meta_graph(load_dir+'.meta')
    loader.restore(sess,load_dir)

    user_id,user_gender,user_age,user_job,movie_categories,movies_title,targets,lr,dropout_keep_prob,_,movie_combine_layer_flat,__=get_tensors(loaded_graph)

    for item in movies.values:
        categories=np.zeros([1,18])
        categories[0]=item.take(2)

        titles=np.zeros([1,sentences_size])
        titles[0]=item.take(1)

        feed={
            movie_id:np.reshape(item.take(0),[1,1]),
            movie_categories:categories,
            movies_title:titles,
            dropout_keep_prob:1
        }

        movie_combine_layer_flat_val=sess.run([movie_combine_layer_flat],feed)
        movie_matrix.append(movie_combine_layer_flat_val)
    
pickle.dump((np.array(movie_matrix).reshape(-1,200)),open('movie_matrix.p','wb'))
movie_matrix=pickle.load(open('movie_matrix.p',mode='rb'))

loaded_graph = tf.Graph()  #
users_matrics = []
with tf.Session(graph=loaded_graph) as sess:  #
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    user_id, user_gender, user_age, user_job, movie_id, movie_categories, movies_title, targets, lr, dropout_keep_prob, _, __,user_combine_layer_flat = get_tensors(loaded_graph)  #loaded_graph

    for item in users.values:
        feed = {
            user_id: np.reshape(item.take(0), [1, 1]),
            user_gender: np.reshape(item.take(1), [1, 1]),
            user_age: np.reshape(item.take(2), [1, 1]),
            user_job: np.reshape(item.take(3), [1, 1]),
            dropout_keep_prob: 1}

        user_combine_layer_flat_val = sess.run([user_combine_layer_flat], feed)  
        users_matrics.append(user_combine_layer_flat_val)

pickle.dump((np.array(users_matrics).reshape(-1, 200)), open('users_matrics.p', 'wb'))
users_matrics = pickle.load(open('users_matrics.p', mode='rb'))

def recommend_same_type_movie(movie_id_val,top_k=20):
    loaded_graph=tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        loader=tf.train.import_meta_graph(load_dir+'.meta')
        loader.restore(sess,load_dir)

        norm_movie_matrix=tf.sqrt(tf.reduce_sum(tf.square(movie_matrix),1,keep_dims=True))
        normalized_movie_matrix=movie_matrix/norm_movie_matrix

        probs_embeddings=(movie_matrix[movieid2idx[movie_id_val]]).reshape([1,200])
        probs_similarity=tf.matmul(probs_embeddings,tf.transpose(normalized_movie_matrix))
        sim=(probs_similarity.eval())
        # results=(-sim[0]).argsort()[0:top_k]
        # print(results)

        print("the movie you want to see is: {}".format(movies_orig[movieid2idx[movie_id_val]]))
        print("there are some recommended for you:")
        p=np.squeeze(sim)
        p[np.argsort(p)[:-top_k]]=0
        p=p/np.sum(p)
        results=set()
        while len(results)!=5:
            c=np.random.choice(3883,1,p=p)[0]
            results.add(c)
        for val in (results):
            print(val)
            print(movies_orig[val])
        
        return results

print(recommend_same_type_movie(1401,20))

def recommend_your_favorite_movie(user_id_val,top_k=10):
    loaded_graph=tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        loader=tf.train.import_meta_graph(load_dir+'.meta')
        loader.restore(sess,load_dir)

        probs_embeddings=(users_matrics[user_id_val-1]).reshape([1,200])

        probs_similarity=tf.matmul(probs_embeddings,tf.transpose(movie_matrix))
        sim=(probs_similarity.eval())
        # print(sim.shape)
        # results=(-sim[0]).argsot()[0:top_k]
        # print(results)
        # sim_norm=probs_norms_similarity.eval()
        # print((-sim_norm[0]).argsort()[0:top_k])

        print("there are some recommended for you:")
        p=np.squeeze(sim)
        p[np.argsort(p)[:-top_k]]=0
        p=p/np.sum(p)
        results=set()
        while len(results)!=5:
            c=np.random.choice(3883,1,p=p)[0]
            results.add(c)
        for val in (results):
            print(val)
            print(movies_orig[val])
        
        return results
print(recommend_your_favorite_movie(234,10))

import random
def recommend_other_favorite_movie(movie_id_val,top_k=20):
    loaded_graph=tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        loader=tf.train.import_meta_graph(load_dir+'.meta')
        loader.restore(sess,load_dir)

        prob_movie_embeddings=(movie_matrix[movieid2idx[movie_id_val]]).reshape([1,200])
        prob_user_favorite_similarity=tf.matmul(prob_movie_embeddings,tf.transpose(users_matrics))
        favorite_user_id=np.argsort(prob_user_favorite_similarity.eval())[0][-top_k:]

        print("the movies you like are: {}".format(movies_orig[movieid2idx[movie_id_val]]))
        
        print("the people who like this movie are： {}".format(users_orig[favorite_user_id-1]))
        prob_users_embeddings=(users_matrics[favorite_user_id-1].reshape([-1,200]))
        prob_similarity=tf.matmul(prob_users_embeddings,tf.transpose(movie_matrix))
        sim=(probs.eval())

        p=np.argmax(sim,1)
        print("the people who like this movie also like:")

        results=set()
        while len(results)!=5:
            c=p[random.randrange(top_k)]
            results.add(c)
        for val in (results):
            print(val)
            print(movies_orig[val])
        return results

print(recommend_other_favorite_movie(1401,20))
'''