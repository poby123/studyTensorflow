import tensorflow as tf

# MNIST 데이터 가져오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 이미지 데이터를 float32로 형변환
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')

# 28 * 28 이미지를 784차원으로 floattening
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])

# [0, 255] 범위의 값을 [0, 1] 범위의 값으로 normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# 레이블 데이터에 one-hot encoding 적용
y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)

# tf.data API를 이용해서 데이터를 섞고 batch 형태로 가져온다.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(60000).batch(100)
train_data_iter = iter(train_data)


# Softmax Regression 모델을 위한 tf.Variable들을 정의
W = tf.Variable(tf.zeros(shape=[784, 10]))
b = tf.Variable(tf.zeros(shape=[10]))


# Softmax Regression 모델을 정의
@tf.function
def softmax_regression(x):
    logits = tf.matmul(x, W) + b
    return tf.nn.softmax(logits)


# Cross-entropy 손실함수를 정의
@tf.function
def cross_entropy_loss(y_pred, y):
    return tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(y_pred), axis=[1]))


# 모델의 정확도
@tf.function
def compute_accuracy(y_pred, y):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


# 최적화를 위한 Gradient-Descent Optimizer
optimizer = tf.optimizers.SGD(0.5)


# 최적화를 위한 function
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = softmax_regression(x)
        loss = cross_entropy_loss(y_pred, y)
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))


# 1000번 반복하면서 최적화 수행
for i in range(1000):
    batch_xs, batch_ys = next(train_data_iter)
    train_step(batch_xs, batch_ys)


# 학습된 모델의 정확도 출력
print('Accuracy : %f' % compute_accuracy(softmax_regression(x_test), y_test)) # 0.91 => 약 91%