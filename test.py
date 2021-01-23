# In my environment, I have to write "activate tensorflow" at command prompt first, then I can import tensorflow. 
import tensorflow as tf

print('The Tensorflow version is : ', tf.__version__) # In my environment, it is 2.4.1
hello = tf.constant('Hello, Tensorflow!')
print(hello)

a = tf.constant(10)
b = tf.constant(12)

print(a + b)
