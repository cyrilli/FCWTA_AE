import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import time
from sklearn.model_selection import train_test_split


def main_test_stacked_denoise_AE(model='relu'):

    
    X_train_data = np.load('./Bearing_data/images.npy')   # 19800, 2048
    y_train_data = np.load('./Bearing_data/labels.npy').reshape(-1) - 1
    X_test = np.load('./Bearing_data/test_images.npy')   # 6000, 2048
    y_test = np.load('./Bearing_data/test_labels.npy').reshape(-1) - 1
    X_train, X_val, y_train, y_val = train_test_split( X_train_data, y_train_data, test_size=0.1, random_state=0 )    

    
    print('X_train.shape', X_train.shape)
    print('y_train.shape', y_train.shape)
    print('X_val.shape', X_val.shape)
    print('y_val.shape', y_val.shape)
    print('X_test.shape', X_test.shape)
    print('y_test.shape', y_test.shape)
    print('X %s   y %s' % (X_test.dtype, y_test.dtype))

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 2048], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

    if model == 'relu':
        act = tf.nn.relu
        act_recon = tf.nn.softplus
    elif model == 'sigmoid':
        act = tf.nn.sigmoid
        act_recon = act

    # Define network
    print("\nBuild Network")
    network = tl.layers.InputLayer(x, name='input_layer')
    # denoise layer for AE
    network = tl.layers.DropoutLayer(network, keep=0.5, name='denoising1')
    # 1st layer
    network = tl.layers.DenseLayer(network, n_units=2048, act = act, name=model+'1')   # number of hidden units
    # the encoded features before being masked for sparsity 
    x_recon1 = network.outputs
    network = tl.layers.LifeTimeSparsityLayer(network, sparsity = 0.1, 
                                              is_train = True, name='life_time_sparse1')    
    recon_layer1 = tl.layers.ReconLayer(network, x_recon=x, n_units=2048,
                                        act = act_recon, name='recon_layer1')
#    # 2nd layer
#    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
#    network = tl.layers.DenseLayer(network, n_units=800, act = act, name=model+'2')
#    recon_layer2 = tl.layers.ReconLayer(network, x_recon=x_recon1, n_units=800,
#                                        act = act_recon, name='recon_layer2')
    # Fully connected layer for classification
    network.outputs = x_recon1
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
    network = tl.layers.DenseLayer(network, n_units=10,
                                            act = tf.identity,
                                            name='output_layer')

    # Define fine-tune process
    y = network.outputs   # (None, 10)
    y_op = tf.argmax(tf.nn.softmax(y), 1)  # (None,)
    cost = tl.cost.cross_entropy(y, y_, name='cost')

    n_epoch = 200
    batch_size = 128
    learning_rate = 0.0001
    print_freq = 10

    train_params = network.all_params

        # train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
    train_op = tf.train.AdamOptimizer(learning_rate , beta1=0.9, beta2=0.999,
        epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

    # Initialize all variables including weights, biases and the variables in train_op
    tl.layers.initialize_global_variables(sess)

    ########################### Pre-train ##################################
    print("\nAll Network Params before pre-train")
    network.print_params()
    print("\nPre-train Layer 1")
    recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val,
                            denoise_name='denoising1', n_epoch=150,
                            batch_size=128, print_freq=10, save=False,
                            save_name='w1pre_')
#    print("\nPre-train Layer 2")
#    recon_layer2.pretrain(sess, x=x, X_train=X_train, X_val=X_val,
#                            denoise_name='denoising1', n_epoch=100,
#                            batch_size=128, print_freq=10, save=False)
    print("\nAll Network Params after pre-train")
    network.print_params()
    ########################### Fine-tune ##################################
    print("\nFine-tune Network")
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(
                                    X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update( network.all_drop )     # enable noise layers
            feed_dict[set_keep['denoising1']] = 1    # disable denoising layer
            sess.run(train_op, feed_dict=feed_dict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(
                                    X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable noise layers
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err
                train_acc += ac
                n_batch += 1
            print("   train loss: %f" % (train_loss/ n_batch))
            print("   train acc: %f" % (train_acc/ n_batch))
            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate.minibatches(
                                        X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable noise layers
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err
                val_acc += ac
                n_batch += 1
            print("   val loss: %f" % (val_loss/ n_batch))
            print("   val acc: %f" % (val_acc/ n_batch))
#            try:
#                # visualize the 1st hidden layer during fine-tune
#                tl.visualize.W(network.all_params[0].eval(), second=10,
#                            saveable=True, shape=[28, 28],
#                            name='w1_'+str(epoch+1), fig_idx=2012)
#            except:
#                raise Exception("# You should change visualize_W(), if you want to save the feature images for different dataset")

    print('Evaluation')
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in tl.iterate.minibatches(
                                X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable noise layers
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err
        test_acc += ac
        n_batch += 1
    print("   test loss: %f" % (test_loss/n_batch))
    print("   test acc: %f" % (test_acc/n_batch))
        # print("   test acc: %f" % np.mean(y_test == sess.run(y_op, feed_dict=feed_dict)))

    # Add ops to save and restore all the variables.
    # ref: https://www.tensorflow.org/versions/r0.8/how_tos/variables/index.html
    saver = tf.train.Saver()
    # you may want to save the model
    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in file: %s" % save_path)
    sess.close()

if __name__ == '__main__':
    sess = tf.InteractiveSession()

    """Stacked Denoising Autoencoder"""
    main_test_stacked_denoise_AE(model='relu')  # model = relu, sigmoid
