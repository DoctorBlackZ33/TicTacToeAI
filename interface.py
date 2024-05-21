
import tensorflow as tf
print("HI")
values = [1, 10, 26.9, 2.8, 166.32, 62.3]
sort_order = tf.argsort(values)
print(sort_order)
