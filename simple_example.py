import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)


# (<tf.Tensor 'Const:0' shape=() dtype=float32>, <tf.Tensor 'Const_1:0' shape=() dtype=float32>)

sess = tf.Session()
print(sess.run([node1, node2]))

# [3.0, 4.0]


node3 = tf.add(node1, node2)
print(sess.run(node3))

node1 = tf.placeholder(tf.float32)
node2 = tf.placeholder(tf.float32)

node3 = tf.add(node1, node2)
print(sess.run(node3, { node1: 3, node2: 5 }))

W = tf.Variable([.1], dtype=tf.float32)
b = tf.Variable([-.1], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

sess.run(tf.global_variables_initializer())
print(sess.run(linear_model, { x: [1,2,3,4,5] }))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, { x: [1,2,3,4,5], y: [1,2,3,4,5] }))
# 47.3

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(tf.global_variables_initializer())
for i in range(3000):
  sess.run(train, { x: [1,2,3,4,5], y: [1,2,3,4,5] })

print(sess.run([W, b]))

# array([ 0.99999988], dtype=float32), array([  3.49248467e-07], dtype=float32)]
