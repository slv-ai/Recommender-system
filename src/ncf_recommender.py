import tensorflow as tf 
import argparse
import os
import numpy as np 
import json

def load_training_data(base_dir):
    df_train=np.load(os.path.join(base_dir,'train.npy'))
    user_train,item_train,y_train=np.split(np.transpose(df_train).flatten(),3)
    return user_train,item_train,y_train

def batch_generator(x,y,batch_size,n_batch,shuffle,user_dim,item_dim):
    user_df,item_df=x
    counter=0
    training_index=np.arange(user_df.shape[0])
    if shuffle:
        np.random.shuffle(training_index)
    while True:
        batch_index=training_index[batch_size*counter : batch_size*counter +1 ]
        user_batch = tf.one_hot(user_df[batch_index],depth=user_dim)
        item_batch = tf.one_hot(item_df[batch_index],depth=item_dim)
        y_batch = y[batch_index]
        counter += 1
        yield [user_batch,item_batch],y_batch

        if counter == n_batch :
            if shuffle:
                np.random.shuffle(training_index)
            counter = 0

def get_user_embedding_layers(inputs,emb_dim):
    user_gmf_emb=tf.keras.layers.Dense(emb_dim,activation='relu')(inputs)
    user_mlp_emb=tf.keras.layers.Dense(emb_dim,activation='relu')(inputs)
    return user_gmf_emb,user_mlp_emb

def get_item_embedding_layers(inputs,emb_dim):
    item_gmf_emb=tf.keras.layers.Dense(emb_dim,activation='relu')(inputs)
    item_mlp_emb=tf.keras.layers.Dense(emb_dim,activation='relu')(inputs)
    return item_gmf_emb,item_mlp_emb

#general matrix factorization branch
def gmf(user_emb,item_emb):
    gmf_mat = tf.keras.layers.Multiply()([user_emb,item_emb])
    return gmf_mat

#multi-layer perceptron branch
def mlp(user_emb,item_emb,dropout_rate):
    def add_layer(dim,input_layer,dropout_rate):
        hidden_layer = tf.keras.layers.Dense(dim,activation='relu')(input_layer)
        if dropout_rate:
            dropout_layer = tf.keras.layers.Dropout(dropout_rate)(hidden_layer)
            return dropout_layer
        return hidden_layer
    concat_layer = tf.keras.layers.Concatenate()([user_emb,item_emb])
    dropout_l1 = tf.keras.layers.Dropout(dropout_rate)(concat_layer)

    dense_layer_1 = add_layer(64,dropout_l1,dropout_rate)
    dense_layer_2 = add_layer(32,dense_layer_1,dropout_rate)
    dense_layer_3 = add_layer(16,dense_layer_2,None)
    dense_layer_4 = add_layer(8.dense_layer_3,None)

    return dense_layer_4
    
