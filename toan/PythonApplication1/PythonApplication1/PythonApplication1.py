import tensorflow as tf

x = tf.Variable(tf.zeros([1,4]))

inc_op = x.assign(tf.add(x,[1.0,1.0,1.0,1.0]))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(10):
    x_text, y = sess.run([x, inc_op])
    print(x_text,y)

