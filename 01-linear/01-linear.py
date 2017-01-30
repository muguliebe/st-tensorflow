import tensorflow as tf

x = [1,2,3]
y = [10,20,30]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# hypothesis
hypothesis = W * x + b

# cost
cost = tf.reduce_mean(tf.square(hypothesis - y))

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
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
            print(step , sess.run(cost) , sess.run(W) , sess.run(b))
