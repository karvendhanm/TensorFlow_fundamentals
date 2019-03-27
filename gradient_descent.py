import numpy as np
import tensorflow as tf

# resetting the default graph
tf.reset_default_graph()

# initializing a variable
x=tf.get_variable("x", shape=(), dtype=tf.float32) # x=tf.get_variable("x", shape=(), dtype=tf.float32, trainable=True)
f=x**2 # f is a tensor

# lets just say we want to minimise f w.r.t to x
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
step = optimizer.minimize(f, var_list=[x]) # step = optimizer.minimize(f)

# all training variables
tf.trainable_variables()

# Making gradient descent steps

# create a session and initialize variables
s=tf.InteractiveSession()
s.run(tf.global_variables_initializer())

# lets us make 10 gradient steps
for i in range(10):
    _, curr_x, curr_f = s.run([step, x, f])
    print("x value: {} y value: {}".format(curr_x, curr_f))

























