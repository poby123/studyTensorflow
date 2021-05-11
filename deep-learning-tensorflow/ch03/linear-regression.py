import tensorflow as tf

# 선형회귀 모델(Wx + b)을 위한 tf.Variable을 선언한다.
W = tf.Variable(tf.random.normal(shape=[1]))
b = tf.Variable(tf.random.normal(shape=[1]))

# 선형회귀 모델을 정의
@tf.function
def linear_model(x):
    return W*x+b

# MSE 손실함수를 정의
@tf.function
def mse_loss(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred - y))

#최적화를 위한 gradient descent optimizer 을 정의
optimizer = tf.optimizers.SGD(0.01)

# 최적화를 위한 function 정의
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = linear_model(x)
        loss = mse_loss(y_pred, y)
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))


# 트레이닝을 위한 입력값과 출력값 준비
x_train = [1, 2, 3, 4]
y_train = [2, 4, 6, 8]

# 경사하강법을 1000번 수행
for _ in range(1000):
    train_step(x_train, y_train)

# 테스트를 위한 입력값 준비
x_test = [3.5, 5, 5.5, 6]

# 테스트 데이터를 이용해서, 학습된 선형회귀 모델이 잘 학습했는지 확인.
# 기댓값 : [7, 10, 11, 12]
print(linear_model(x_test).numpy())