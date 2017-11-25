## Semantic Segmentation

#### Kemal Tepe, ketepe@gmail.command

### Introduction
In this project, the pixels of a road in images are labeled using a Fully Convolutional Network (FCN).

### Summary and Objectives

Objective of the project is to design a FCN network to provide segmentation using labeled images from [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php). FCN is constructed using [Python 3](https://www.python.org/) with  [TensorFlow](https://www.tensorflow.org/) libraries (> version 1.0). The fully trained CCN network based on  [VGG-16]('https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip) is used as the CCN architecture. The project uses this architecture and reconstructs the decoder to complete the FCN for segmentation.

#### Objectives:
* Obtain the pre-trained VGG-16 architecture.
* Build the decoding layer.
* Train and optimize the decoding layer to complete the FCN.
* Demonstrate the FCN works.

#### Obtain the pre-trained VGG-16 architecture

The trained VGG is obtained in from [here]('https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip). The vgg layer information and coefficient are stored to the directory pointed by ```vgg_path``` variable. Then the model is loaded using the following command of the TensorFlow (tf).

```Python
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    # get the layer of vgg
    graph =tf.get_default_graph()
    vgg_input= graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep= graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input, vgg_keep, vgg_layer3, vgg_layer4, vgg_layer7

```

#### Build the decoding layer
Decoding layer reverses the CNN to provide pixel level identification for segmentation. With that, we can classify objects in the image for autonomous driving for example in this project, roads are identified such that the vehicles drive on.

The decoding is performed by using the following python function.

```Python
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # FCN-Decoders
    # skip things lecture 10 scene understanding.
    layer7_conv1x1=tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                    kernel_initializer=tf.random_normal_initializer(stddev=STDDEV),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    #upsample layer 7
    layer4_input1 = tf.layers.conv2d_transpose(layer7_conv1x1, num_classes, 4, 2, padding='same',
                                               kernel_initializer=tf.random_normal_initializer(stddev=STDDEV),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    layer4_input2= tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                    kernel_initializer=tf.random_normal_initializer(stddev=STDDEV),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    #skip connection
    layer4_conv1x1=tf.add(layer4_input1, layer4_input2)
    #
    layer3_input1=tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                   kernel_initializer=tf.random_normal_initializer(stddev=STDDEV),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    #upsample layer 4
    layer3_input2=tf.layers.conv2d_transpose(layer4_conv1x1, num_classes, 4, 2, padding='same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=STDDEV),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    layer3_conv1x1=tf.add(layer3_input1, layer3_input2)

    #last layer
    layer_out=tf.layers.conv2d_transpose(layer3_conv1x1, num_classes, 16, 8, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=STDDEV),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return layer_out
```
Hyper parameters such as STDDEV and Regularizer values are varied to reduce the cross entropy loss.

#### Train and optimize the decoding layer to complete the FCN
Training the decoder are done using AdamOptimizer by varying hyper parameters of number of epochs and learning rate. With that, the objective is to reduce the loss (error) by chancing model connection coefficients.

Following function provides the training.

```Pyhton
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    #done
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        for image, label in get_batches_fn(batch_size):
            #_, loss= sess.run([train_op, cross_entropy_loss],
            #            feed_dict={input_image: image, correct_label: label, keep_prob: keep_prob, learning_rate: learning_rate})
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict = {input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0002})

            print("Epoch {} of {}  ".format(epoch+1, epochs), " Loss: {:.4f} ".format(loss))
        print()
```

And the optimizer used in the training is given by

```Pyhton
def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    #done
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # now define a loss function and a trainer/optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return logits, optimizer, loss
```

#### Demonstrate the FCN works.

The FCN implementation is run with different hyper parameters, namely learning_rate, number of epochs, standard deviation and l2_regularizer values. The trade-off in the system are as follows. Increased number of epochs improves the loss hence decreases the errors, however, the time it takes to complete the training also increases. Learning curve has similar impact, Smaller learning curve values requires more time to converge to a stable loss value but the final loss would be lower. The number of epochs and learning rate needs to be carefully adjusted to obtain the best possible model with smaller number of epochs.


Reference-style:
![alt text](./samples/sample1.png)

![alt text][./samples/sample2.png]

![alt text][./samples/sample3.png]


### Conclusions
