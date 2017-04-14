import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import time
from sklearn.model_selection import train_test_split
import os

class Model(object):
    """

    """

    def __init__(self):
        pass

    @staticmethod
    def start_new_session(sess):
        saver = tf.train.Saver()  # create a saver

        sess.run(tf.global_variables_initializer())
        print('started a new session')
        return saver

    @staticmethod
    def continue_previous_session(sess, ckpt_file):
        saver = tf.train.Saver()  # create a saver

        with open(ckpt_file) as file:  # read checkpoint file
            line = file.readline()  # read the first line, which contains the file name of the latest checkpoint
            ckpt = line.split('"')[1]

        # restore
        saver.restore(sess, './log/saver/'+ckpt)
        print('restored from checkpoint ' + ckpt)
        return saver

class denoise_AE():
    def __init__(self):
        # placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, 2048], name='x')
        self.y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')
    
        print("Build Network")
        self.network = tl.layers.InputLayer(self.x, name='input_layer')
        self.network = tl.layers.DropoutLayer(self.network, keep=0.5, name='denoising1')    # if drop some inputs, it is denoise AE
        self.network = tl.layers.DenseLayer(self.network, n_units=2048,
                                    act = tf.nn.relu, name='relu1')
        self.x_recon1 = self.network.outputs
        self.network = tl.layers.LifeTimeSparsityLayer(self.network, sparsity = 0.1, 
                                                  is_train = True, name='life_time_sparse')
        self.recon_layer1 = tl.layers.ReconLayer(self.network, x_recon=self.x, n_units=2048,
                                    act = tf.nn.softplus, name='recon_layer1')

def train(X_train, X_val, new_training = True):    
#    ## ready to train
#    tl.layers.initialize_global_variables(sess)
    AE = denoise_AE()

    with tf.Session() as sess:
        if new_training:
            saver = Model.start_new_session(sess)
        else:
            saver = Model.continue_previous_session(sess, ckpt_file='./log/saver/checkpoint')
        ## print all params
        print("All Network Params")
        AE.network.print_params()    
        ## pretrain
        print("Pre-train Layer 1")
        AE.recon_layer1.pretrain(sess, x=AE.x, X_train=X_train, X_val=X_val,
                                 denoise_name='denoising1', n_epoch=200,
                                 batch_size=128, print_freq=10, save=False, checkpoint_saver = saver)
    # You can also disable denoisong by setting denoise_name=None.
    # recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val,
    #                           denoise_name=None, n_epoch=500, batch_size=128,
    #                           print_freq=10, save=True, save_name='w1pre_')

def test(data):
    AE = denoise_AE()
    with tf.Session() as sess:
        saver = Model.continue_previous_session(sess, ckpt_file='./log/saver/checkpoint')
        ## print all params
        print("All Network Params")
        AE.network.print_params()
        
        feed_dict = {AE.x : data}
        dp_dict = tl.utils.dict_to_one( AE.network.all_drop )  # disable noise layers
        feed_dict.update(dp_dict)
        encode_ = sess.run(AE.x_recon1, feed_dict)
        print('extracted features shape: ', encode_.shape)
        np.save('./Bearing_data/extracted_features/X_test_encode.npy', encode_)

if __name__ == '__main__':

    X_train_data = np.load('./Bearing_data/images.npy')   # 19800, 2048
    y_train_data = np.load('./Bearing_data/labels.npy').reshape(-1)
    X_test = np.load('./Bearing_data/test_images.npy')   # 6000, 2048
    y_test = np.load('./Bearing_data/test_labels.npy').reshape(-1)

    X_train, X_val, y_train, y_val = train_test_split( X_train_data, y_train_data, test_size=0.1, random_state=0 )

    print('X_train.shape', X_train.shape)
    print('y_train.shape', y_train.shape)
    print('X_val.shape', X_val.shape)
    print('y_val.shape', y_val.shape)
    print('X_test.shape', X_test.shape)
    print('y_test.shape', y_test.shape)
    print('X %s   y %s' % (X_test.dtype, y_test.dtype))
    
    #train(X_train, X_val, new_training = True)
    test(X_test)