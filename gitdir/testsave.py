import tensorflow as tf
import os

X = tf.placeholder(tf.float32,[None,1])

W = tf.Variable(tf.zeros([1,1]))

B = tf.Variable(tf.zeros([1]))

y = tf.matmul(X,W) + B

y_ = tf.placeholder(tf.float32,[None,1])

cost = tf.reduce_mean(tf.square(y-y_))

opt = tf.train.AdagradOptimizer(1).minimize(cost)

feed = {X:[[1],[4]], y_:[[3],[1]]}

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    #sess.run(init)
    saver.restore(sess,"testDir/model.ckpt-97")
    for i in range(100):
        _, cost_ = sess.run([opt,cost], feed_dict=feed)
        print cost_
        saver.save(sess, os.path.join("testDir", 'model.ckpt'), global_step=i)
