import tensorflow as tf

hello = tf.constant('hello, tf')
sess = tf.Session()

print(sess.run(hello))
print('hello2' , 'a', 'c')




