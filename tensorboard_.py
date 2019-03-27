import numpy as np
import tensorflow as tf

tf.reset_default_graph()

x=tf.get_variable('x', shape=(), dtype=np.float32)
f=x**2

optimizer = tf.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(f,var_list=[x])

# Logging with tensorboard
tf.summary.scalar("curr_x",x)
tf.summary.scalar("curr_f",f)
summaries = tf.summary.merge_all()

# logging the summaries:
s=tf.InteractiveSession()
summary_writer = tf.summary.FileWriter("logs/1",s.graph)
s.run(tf.global_variables_initializer())

for i in range(10):
    _, curr_summaries = s.run([step, summaries])
    summary_writer.add_summary(curr_summaries, i)
    summary_writer.flush()
