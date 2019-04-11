'''
Few peculiarities of tensorflow. This is a define-and-run type of programming.
However pytorch is a define-by-run programming where the chainer memorizes the history of computation
instead of programming logic thereby making it more intuitive.

In tensorflow

step1: Assemble the computation graph
step2: Interact with the said computation graph using Tensorflow's sessions.

what is a computation graph?
Essentially, it's a global data structure: A directed graph that captures instructions about how to calculate things.

'''

########
# Example 1

import tensorflow as tf
two_node = tf.constant(2) # creates a node which contains the constant 2
print(two_node) # this line prints a tensor which is a pointer to the aforementioned node.

########

########
# Example 2

import tensorflow as tf
two_node = tf.constant(2)
print(two_node)
another_two_node = tf.constant(2)
print(another_two_node)
two_node = tf.constant(2)
print(two_node)
tf.constant(2)
########

########

# Example 3
import tensorflow as tf
two_node = tf.constant(2)
print(two_node)
another_two_node = two_node
print(another_two_node)
two_node = None
print(two_node)
########

########
import tensorflow as tf
two_node = tf.constant(2)
print(two_node)
three_node = tf.constant(3)
print(three_node)
sum_node = two_node + three_node # equivalent to tf.add(two_node, three_node) also + is overloaded as it adds a new node to the graph
print(sum_node)# Computational graphs contain only the steps of computation, they do not contain the results

########

########

# Session allows you to handle memory allocation and optimization that allows us to actually perform the computations
import tensorflow as tf
two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node
sess = tf.Session()
print(sess.run(sum_node))

########

########

import tensorflow as tf
two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node
sess = tf.Session()
print(sess.run([two_node, sum_node]))

########

########
# Placeholders & feed_dict

import tensorflow as tf
input_placeholder = tf.placeholder(tf.int32)
sess = tf.Session()
print(sess.run(input_placeholder)) # input_placeholder is still just a placeholder.
print(sess.run(input_placeholder, feed_dict={input_placeholder:2}))

########

########

#Third Key Abstraction:  Computation Paths

import tensorflow as tf
input_placeholder = tf.placeholder(tf.int32)
three_node = tf.constant(3)
sum_node = input_placeholder + three_node
sess = tf.Session()
print(sess.run(three_node))
print(sess.run(sum_node))
print(sess.run(sum_node, feed_dict={input_placeholder : 2}))

########

########
# Variables and side effects

import tensorflow as tf
count_variable = tf.get_variable("count",[])
print(count_variable)
sess = tf.Session()
print(sess.run(count_variable))

########

########

# closer look at tf.assign

import tensorflow as tf
count_variable = tf.get_variable("count",[])
zero_node = tf.constant(0.)
assign_node = tf.assign(count_variable, zero_node)
sess = tf.Session()
sess.run(assign_node)
sess.run(count_variable)

########

########

# Third Key Abstraction: Computation Paths
import tensorflow as tf
input_placeholder = tf.placeholder(tf.int32)
three_node = tf.constant(2)
sum_node = input_placeholder + three_node
sess = tf.Session()
sess.run(three_node)
sess.run(sum_node)
sess.run(sum_node, feed_dict={input_placeholder : 3})

########

########

import tensorflow as tf
input_placeholder = tf.placeholder(tf.int32)
three_node = tf.constant(3)
sess = tf.Session()
sess.run(three_node)
sum_node = input_placeholder + three_node
sess.run(sum_node)
sess.run(sum_node, feed_dict={input_placeholder:7})

########

########
# Variables & Side Effects
# We've seen two types of 'no-ancestor' nodes. tf.constant and tf.placeholder.
# tf.Variable() is the older version of tf.get_variable()

import tensorflow as tf
# tf.variable_scope()
count_variable = tf.get_variable("count", [])
sess = tf.Session()
sess.run(count_variable)

########

########
import tensorflow as tf
count_variable = tf.get_variable("count", [])
zero_node = tf.constant(3.)
assign_node = tf.assign(count_variable, zero_node)
sess = tf.Session()
sess.run(assign_node)
sess.run(count_variable)
sess.run(zero_node)
########


########
# Initializers
import tensorflow as tf
const_init_node = tf.constant_initializer(0.)
# const_init_node = tf.constant(0.)
count_variable = tf.get_variable("count", [], initializer=const_init_node)
sess = tf.Session()
sess.run(const_init_node)
print(count_variable)
sess.run([count_variable])
sess.run(count_variable)

########

########

import tensorflow as tf
const_init_node = tf.constant_initializer(0.)
count_variable = tf.get_variable('count', [], initializer=const_init_node)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sess.run([count_variable])
sess.run(count_variable)

########

########

### two ways to assign a value to a variable 1) assign and 2) using initializers
## using assign

import tensorflow as tf
count_variable = tf.get_variable('count', [])
zero_node = tf.constant(0.)
assign_node = tf.assign(count_variable, zero_node)
sess = tf.Session()
sess.run(assign_node)
sess.run(count_variable)
sess.run(zero_node)

## using initializer

import tensorflow as tf
const_init = tf.constant_initializer(0.)
count_variable = tf.get_variable('count', [], initializer=const_init)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sess.run(count_variable)

########

########
# Toy linear regression problem

import tensorflow as tf

### build the graph
## first set up the parameters
m = tf.get_variable('m',[], initializer=tf.constant_initializer(0.))
b = tf.get_variable('b',[], initializer=tf.constant_initializer(0.))
init = tf.global_variables_initializer()

## then let us set up the computations
input_placeholder = tf.placeholder(tf.float32)
output_placeholder = tf.placeholder(tf.float32)

x = input_placeholder
y = output_placeholder

y_guess = m * x + b

loss = tf.square(y - y_guess)

## finally, set up the optimizer and minimization node
optimizer = tf.train.GradientDescentOptimizer(1e-3)
train_op = optimizer.minimize(loss)

### start the session
sess = tf.Session()
sess.run(init)

### perform the training loop
import random

# set up the problem
true_m = random.random()
true_b = random.random()

for update_i in range(100000):
    input_data = random.random()
    output_data = true_m * input_data + true_b

    _loss, _ = sess.run([loss, train_op], feed_dict={input_placeholder:input_data, output_placeholder:output_data})
    print(update_i, _loss)

print("True parameters:     m=%.4f, b=%.4f" % (true_m, true_b))
print("Learned parameters:  m=%.4f, b=%.4f" % tuple(sess.run([m, b])))

########

########
# Debugging with tf.Print

import tensorflow as tf
two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node
sess = tf.Session()
print(sess.run([two_node, three_node]))
print(sess.run(sum_node))

########

########

# Alternatives to tf.Print to see intermediatery values

import tensorflow as tf
two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node
sess = tf.Session()
answer, inspection = sess.run([sum_node, [two_node, three_node]])
print(inspection)
print(answer)
inspection, answer = sess.run([[two_node, three_node], sum_node])
print(inspection)
print(answer)

########
































