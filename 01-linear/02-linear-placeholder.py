import tensorflow as tf

x_data = [1,2,3]
y_data = [10,20,30]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# v
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# hypothesis
hypothesis = W * X + b

# cost
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
a = tf.Variable(0.1) # learning rate, alpah
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# init
init = tf.initialize_all_variables()

# launch the graph
sess = tf.Session()
sess.run(init)

# Fit
for step in range(1000):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
            print(step , sess.run(cost, feed_dict={X:x_data, Y:y_data}) , sess.run(W) , sess.run(b))

# and.. now..
print("when 5 =>", sess.run(hypothesis, feed_dict={X:5}))
print("when 3 =>", sess.run(hypothesis, feed_dict={X:3}))
