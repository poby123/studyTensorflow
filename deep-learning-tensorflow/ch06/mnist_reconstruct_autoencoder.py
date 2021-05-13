# 출처: https://github.com/solaris33/deep-learning-tensorflow-book-code/blob/master/Ch06-AutoEncoder/mnist_reconstruction_using_autoencoder_v2.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# MNIST 데이터를 다운로드 합니다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 이미지들을 float32 데이터 타입으로 변경합니다.
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')

# 28*28 형태의 이미지를 784차원으로 flattening 합니다.
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])

# [0, 255] 사이의 값을 [0, 1]사이의 값으로 Normalize합니다.
x_train, x_test = x_train / 255., x_test / 255.

# 학습에 필요한 설정값들을 정의합니다.
learning_rate = 0.02
training_epochs = 50    # 반복횟수
batch_size = 256        # 배치개수
display_step = 1        # 손실함수 출력 주기
examples_to_show = 10   # 보여줄 MNIST Reconstruction 이미지 개수
input_size = 784        # 28*28
hidden1_size = 256 
hidden2_size = 128

# tf.data API를 이용해서 데이터를 섞고 batch 형태로 가져옵니다.
train_data = tf.data.Dataset.from_tensor_slices(x_train)
train_data = train_data.shuffle(60000).batch(batch_size)

# Autoencoder 모델을 정의합니다.
class AutoEncoder(object):

  # Autoencoder 모델을 위한 tf.Variable들을 정의합니다.
  def __init__(self):
    # 인코딩(Encoding) - 784 -> 256 -> 128
    self.W1 = tf.Variable(tf.random.normal(shape=[input_size, hidden1_size]))
    self.b1 = tf.Variable(tf.random.normal(shape=[hidden1_size]))
    self.W2 = tf.Variable(tf.random.normal(shape=[hidden1_size, hidden2_size]))
    self.b2 = tf.Variable(tf.random.normal(shape=[hidden2_size]))
    
    # 디코딩(Decoding) 128 -> 256 -> 784
    self.W3 = tf.Variable(tf.random.normal(shape=[hidden2_size, hidden1_size]))
    self.b3 = tf.Variable(tf.random.normal(shape=[hidden1_size]))
    self.W4 = tf.Variable(tf.random.normal(shape=[hidden1_size, input_size]))
    self.b4 = tf.Variable(tf.random.normal(shape=[input_size]))

  def __call__(self, x):
    H1_output = tf.nn.sigmoid(tf.matmul(x, self.W1) + self.b1)
    H2_output = tf.nn.sigmoid(tf.matmul(H1_output, self.W2) + self.b2)
    H3_output = tf.nn.sigmoid(tf.matmul(H2_output, self.W3) + self.b3)
    reconstructed_x = tf.nn.sigmoid(tf.matmul(H3_output, self.W4) + self.b4)

    return reconstructed_x

# MSE 손실 함수를 정의합니다.
@tf.function
def mse_loss(y_pred, y_true):
  return tf.reduce_mean(tf.pow(y_true - y_pred, 2)) # MSE(Mean of Squared Error) 손실함수

# 최적화를 위한 RMSProp 옵티마이저를 정의합니다.
optimizer = tf.optimizers.RMSprop(learning_rate)

# 최적화를 위한 function을 정의합니다.
@tf.function
def train_step(model, x):
  # 타겟데이터는 인풋데이터와 같습니다.
  y_true = x
  with tf.GradientTape() as tape:
    y_pred = model(x)
    loss = mse_loss(y_pred, y_true)
  gradients = tape.gradient(loss, vars(model).values())
  optimizer.apply_gradients(zip(gradients, vars(model).values()))

# Autoencoder 모델을 선언합니다.
AutoEncoder_model = AutoEncoder()


# 지정된 횟수만큼 최적화를 수행합니다.
for epoch in range(training_epochs):
  # 모든 배치들에 대해서 최적화를 수행합니다.
  # Autoencoder는 Unsupervised Learning이므로 타겟 레이블(label) y가 필요하지 않습니다.
  for batch_x in train_data:
    # 옵티마이저를 실행해서 파라마터들을 업데이트합니다.
    _, current_loss = train_step(AutoEncoder_model, batch_x), mse_loss(AutoEncoder_model(batch_x), batch_x)
  
  # 지정된 epoch마다 학습결과를 출력합니다.
  if epoch % display_step == 0:
    print("반복(Epoch): %d, 손실 함수(Loss): %f" % ((epoch+1), current_loss))


# 테스트 데이터로 Reconstruction을 수행합니다.
reconstructed_result = AutoEncoder_model(x_test[:examples_to_show])

# 원본 MNIST 데이터와 Reconstruction 결과를 비교합니다.
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
  a[0][i].imshow(np.reshape(x_test[i], (28, 28)))
  a[1][i].imshow(np.reshape(reconstructed_result[i], (28, 28)))

f.savefig('reconstructed_mnist_image.png')  # reconstruction 결과를 png로 저장합니다.
f.show()
plt.draw()
plt.waitforbuttonpress()

# 결과
'''
반복(Epoch): 1, 손실 함수(Loss): 0.062777
반복(Epoch): 2, 손실 함수(Loss): 0.050200
반복(Epoch): 3, 손실 함수(Loss): 0.044351
반복(Epoch): 4, 손실 함수(Loss): 0.040442
반복(Epoch): 5, 손실 함수(Loss): 0.042560
반복(Epoch): 6, 손실 함수(Loss): 0.040495
반복(Epoch): 7, 손실 함수(Loss): 0.038475
반복(Epoch): 8, 손실 함수(Loss): 0.034478
반복(Epoch): 9, 손실 함수(Loss): 0.037236
반복(Epoch): 10, 손실 함수(Loss): 0.034368
반복(Epoch): 11, 손실 함수(Loss): 0.031964
반복(Epoch): 12, 손실 함수(Loss): 0.034743
반복(Epoch): 13, 손실 함수(Loss): 0.028969
반복(Epoch): 14, 손실 함수(Loss): 0.034610
반복(Epoch): 15, 손실 함수(Loss): 0.030000
반복(Epoch): 16, 손실 함수(Loss): 0.027104
반복(Epoch): 17, 손실 함수(Loss): 0.026887
반복(Epoch): 18, 손실 함수(Loss): 0.027279
반복(Epoch): 19, 손실 함수(Loss): 0.022892
반복(Epoch): 20, 손실 함수(Loss): 0.026611
반복(Epoch): 21, 손실 함수(Loss): 0.026551
반복(Epoch): 22, 손실 함수(Loss): 0.026881
반복(Epoch): 23, 손실 함수(Loss): 0.024525
반복(Epoch): 24, 손실 함수(Loss): 0.027696
반복(Epoch): 25, 손실 함수(Loss): 0.024122
반복(Epoch): 26, 손실 함수(Loss): 0.023690
반복(Epoch): 27, 손실 함수(Loss): 0.029098
반복(Epoch): 28, 손실 함수(Loss): 0.028002
반복(Epoch): 29, 손실 함수(Loss): 0.025415
반복(Epoch): 30, 손실 함수(Loss): 0.022848
반복(Epoch): 31, 손실 함수(Loss): 0.025457
반복(Epoch): 32, 손실 함수(Loss): 0.025138
반복(Epoch): 33, 손실 함수(Loss): 0.021410
반복(Epoch): 34, 손실 함수(Loss): 0.021874
반복(Epoch): 35, 손실 함수(Loss): 0.022993
반복(Epoch): 36, 손실 함수(Loss): 0.023085
반복(Epoch): 37, 손실 함수(Loss): 0.023549
반복(Epoch): 38, 손실 함수(Loss): 0.023941
반복(Epoch): 39, 손실 함수(Loss): 0.026214
반복(Epoch): 40, 손실 함수(Loss): 0.023420
반복(Epoch): 41, 손실 함수(Loss): 0.023365
반복(Epoch): 42, 손실 함수(Loss): 0.022887
반복(Epoch): 43, 손실 함수(Loss): 0.024564
반복(Epoch): 44, 손실 함수(Loss): 0.023213
반복(Epoch): 45, 손실 함수(Loss): 0.021106
반복(Epoch): 46, 손실 함수(Loss): 0.023394
반복(Epoch): 47, 손실 함수(Loss): 0.020840
반복(Epoch): 48, 손실 함수(Loss): 0.023499
반복(Epoch): 49, 손실 함수(Loss): 0.019049
반복(Epoch): 50, 손실 함수(Loss): 0.022492
'''