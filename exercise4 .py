
from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data import Data

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    return conv_2d

def deconv2d(x, filters):
    return tf.layers.conv2d_transpose(x,
	                              filters=filters, 
                                  kernel_size=[2, 2],
								  strides=[2, 2], 
								  padding='VALID')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def crop_and_concat(layer, inx, target):
    offset = (inx.get_shape()[1] - target)//2
    x_crop = inx[:,
                 offset:target + offset,
                 offset:target + offset
                 :]
    return tf.concat([layer, x_crop], axis=3)   


def downlayers(input, inputfeatures, outfeatures):
    # 2 covolutional layers and max pooling with filter size 3x3
	# 1st convolution
    shape1 = [3, 3, inputfeatures, outfeatures]
    w1 = weight_variable(shape1)
    b1 = bias_variable(shape=[outfeatures])
    conv1 = conv2d(input, w1)
    tconv1 = tf.nn.relu(conv1 + b1)
	# 2nd convolution
    shape2 = [3, 3, outfeatures, outfeatures]
    w2 = weight_variable(shape2)
    b2 = bias_variable(shape=[outfeatures])
    conv2 = conv2d(tconv1, w2)
    layer = conv2 + b2
    tconv2 = tf.nn.relu(layer)
        
    outx = max_pool(tconv2)
    return outx, layer
    
    
def extralayers(input, inputfeatures, outfeatures):
    # 2 covolutional layers

	# 1st convolution
    shape1 = [3, 3, inputfeatures, outfeatures]
    w1 = weight_variable(shape1)
    b1 = bias_variable(shape=[outfeatures])
    conv1 = conv2d(input, w1)
    tconv1 = tf.nn.relu(conv1 + b1)
	# 2nd convolution
    shape2 = [3, 3, outfeatures, outfeatures]
    w2 = weight_variable(shape2)
    b2 = bias_variable(shape=[outfeatures])
    conv2 = conv2d(tconv1, w2)
    tconv2 = tf.nn.relu(conv2 + b2)
    return tconv2
        
def uplayers(input, x_to_crop, inputfeatures, outfeatures):
    in_x = deconv2d(input, outfeatures)
    concat = crop_and_concat(in_x, x_to_crop, in_x.get_shape()[1])
	# 1st convolution
    shape1 = [3, 3, inputfeatures, outfeatures]
    w1 = weight_variable(shape1)
    b1 = bias_variable(shape=[outfeatures])
    conv1 = conv2d(concat, w1)
    tconv1 = tf.nn.relu(conv1 + b1)
	# 2nd convolution
    shape2 = [3, 3, outfeatures, outfeatures]
    w2 = weight_variable(shape2)
    b2 = bias_variable(shape=[outfeatures])
    conv2 = conv2d(tconv1, w2)
    tconv2 = tf.nn.relu(conv2 + b2)
    return tconv2
	

def output_map(input, inputfeatures, outfeatures):
    shape = [1, 1, inputfeatures, outfeatures]
    w = weight_variable(shape)
    b = bias_variable(shape=[outfeatures])
    conv = conv2d(input, w)
    tconv = conv + b
    return tconv
	
def accuracy(x, data, i, sess, output):
    # training accuracy
    image, labels = data.get_train_image_list_and_label_list()
    out = sess.run(output, feed_dict={x: image})
    prediction =  np.argmax(out, axis=3)
    correct_pix = np.sum(prediction == labels)
	# how many images x size of the image
    total_pix = len(labels) * labels[0].shape[0] * \
            labels[0].shape[1]
    incorrect_pix = np.sum(prediction != labels)
    train_acc = correct_pix / (total_pix + incorrect_pix)
    print('Iteration %i: %1.4f' % (i, train_acc))
	
	# test accuracy
    test_image, test_labels = data.get_test_image_list_and_label_list()
    test_out = sess.run(output, feed_dict={x: test_image})
    test_prediction =  np.argmax(test_out, axis=3)
    test_correct_pix = np.sum(test_prediction == test_labels)
    # how many images x size of a image
    test_total_pix = len(test_labels) * test_labels[0].shape[0] * \
            test_labels[0].shape[1]
    test_incorrect_pix = np.sum(test_prediction != test_labels)
    test_acc = test_correct_pix / (test_total_pix + test_incorrect_pix)
    return train_acc, test_acc

def plot_image(im):
    figure = plt.figure()
    ax = plt.Axes(figure, [0., 0., 1., 1.])
    figure.add_axes(ax)
    ax.imshow(im, cmap='gray')
    plt.show()
	
	

def main():
    n_class = 2
    d = Data()
    inputres = 300
    outres = 116
    y = tf.placeholder(tf.int64, shape=[None, outres, outres])
    x = tf.placeholder(tf.float32, shape=[None, inputres, inputres, 1])
	
	# 4 downsampling layers, contain 2 convolutions and max pooling
    down0, f0 = downlayers(x,inputfeatures = 1, outfeatures = 32)
    down1, f1 = downlayers(down0,inputfeatures = 32, outfeatures = 64)
    down2, f2 = downlayers(down1,inputfeatures = 64, outfeatures = 128)
    down3, f3 = downlayers(down2,inputfeatures = 128, outfeatures = 256)
	
	# 2 extra convolutional layers, no max pooling
    extra_x = extralayers(down3,inputfeatures = 256, outfeatures = 512)
	
	# 4 upsampling layers, contain 2 convolutions and one transposed convolution
    up0 = uplayers(extra_x, f3, inputfeatures = 512, outfeatures = 256)
    up1 = uplayers(up0, f2, inputfeatures = 256, outfeatures = 128)
    up2 = uplayers(up1, f1, inputfeatures = 128, outfeatures = 64)
    up3 = uplayers(up2, f0, inputfeatures = 64, outfeatures = 32)
	
    # last layer
    last = output_map(up3, inputfeatures = 32, outfeatures = n_class)
    # softmax output
    output = tf.nn.softmax(last)
    # loss with cross entropy
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=last)

    train_step = tf.train.AdamOptimizer(0.0001, 0.95, 0.99).minimize(loss)

    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_steps = 40000
            epochs = []
            train_acc = []
            test_acc = []
            for i in range(num_steps + 1):
                if i%100 == 0:
			        # store accuracy
                    epochs.append(i)
                    train, test = accuracy(x, d, i, sess, output)
                    train_acc.append(train)
                    test_acc.append(test)
                    batchx, batchy = d.get_train_image_list_and_label_list()
                # Run optimization op (backprop)
                o = sess.run([train_step], feed_dict={x: batchx, y: batchy})
				
            test_im, _ = d.get_test_image_list_and_label_list()

            for i in range(2):
                # plot original images
                image = test_im[i]
                im = np.array([[p[0] for p in l] for l in image])
                plot_image(im)
                # plot segmented images
                test_out = sess.run(output, feed_dict={x: test_im})
                test_prediction =  np.argmax(test_out, axis=3)
                plot_image(test_prediction[i])

    # plotting the accuracy
    plt.plot(epochs, train_acc, label='train')
    plt.plot(epochs, test_acc, label='test')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.show()
    

if __name__ == '__main__':
    main()


