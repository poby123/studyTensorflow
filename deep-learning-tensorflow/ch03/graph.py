import tensorflow as tf
print(tf.__version__)  # 2.4.1

# define nodes
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0, dtype=tf.float32) # 여기서 dtype은 생략가능하다.
node3 = node1 + node2

# tf.Tensor(3.0, shape=(), dtype=float32) tf.Tensor(4.0, shape=(), dtype=float32)
print(node1, node2)

# 3.0 4.0  
print(node1.numpy(), node2.numpy())

# ============================

# node3 :  tf.Tensor(7.0, shape=(), dtype=float32)
print("node3 : ", node3)

# 7.0
print(node3.numpy())
