# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 19:02:09 2018

@author: Kirk
"""

import tensorflow as tf

sess = tf.Session()

q = tf.FIFOQueue(3, "float")
init = q.enqueue_many(([0.,0.,0.],))

x = q.dequeue()
y = x+1
q_inc = q.enqueue([y])

init.run()
q_inc.run()