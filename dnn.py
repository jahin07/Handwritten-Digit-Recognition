import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

hLayer1 = 1500
hLayer2 = 1500
hLayer3 = 1500
hLayer4 = 1500

classes = 10
batch_size = 100 #100 features nad feed them to our network. CHnage to 1000

x = tf.placeholder('float',[None, 784]) #input
y = tf.placeholder('float') #label



def neural_network_model(data):
    hl1 = {'weights':tf.Variable(tf.random_normal([784, hLayer1])), 'biases':tf.Variable(tf.random_normal([hLayer1]))}
    
    hl2 = {'weights':tf.Variable(tf.random_normal([hLayer1, hLayer2])), 'biases':tf.Variable(tf.random_normal([hLayer2]))}
    
    hl3 = {'weights':tf.Variable(tf.random_normal([hLayer2, hLayer3])), 'biases':tf.Variable(tf.random_normal([hLayer3]))}

    hl4 = {'weights':tf.Variable(tf.random_normal([hLayer3, hLayer4])), 'biases':tf.Variable(tf.random_normal([hLayer4]))}
    
    opLayer = {'weights':tf.Variable(tf.random_normal([hLayer4, classes])), 'biases':tf.Variable(tf.random_normal([classes]))}

    #(input_data * weight) + biases
    
    layer1 = tf.add(tf.matmul(data, hl1['weights']), hl1['biases'])
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1, hl2['weights']), hl2['biases'])
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.add(tf.matmul(layer2, hl3['weights']), hl3['biases'])
    layer3 = tf.nn.relu(layer3)

    layer4 = tf.add(tf.matmul(layer3, hl4['weights']), hl4['biases'])
    layer4 = tf.nn.relu(layer4)

    output = tf.matmul(layer4, opLayer['weights']) + opLayer['biases']

    return output

def train_neural_network(x):
    predict = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
    optimize = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range (int (mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimize, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss+=c

            print'Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss

        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print 'Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels})*100

train_neural_network(x)
