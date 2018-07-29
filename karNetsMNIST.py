
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[2]:


mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


# In[4]:


x = tf.placeholder(tf.float32,shape=[None,784])
y_ = tf.placeholder(tf.float32,[None,10])


# In[5]:


#define weights and balances
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# In[8]:


y = tf.nn.softmax(tf.matmul(x,W)+b)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(                              labels=y_,logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys})

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy= tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

test_accuracy = sess.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels})
print("Test Accuracy ={0}% ".format(test_accuracy*100.0))
sess.close()

