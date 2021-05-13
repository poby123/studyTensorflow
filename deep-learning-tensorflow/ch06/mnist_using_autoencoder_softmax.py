# 출처 : https://github.com/solaris33/deep-learning-tensorflow-book-code/blob/master/Ch06-AutoEncoder/mnist_classification_using_autoencoder_and_softmax_classifier_v2.py
# MNIST 숫자 분류를 위한 Autoencoder+Softmax 분류기 예제 

import tensorflow as tf

# MNIST 데이터를 다운로드 합니다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 이미지들을 float32 데이터 타입으로 변경합니다.
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')

# 28*28 형태의 이미지를 784차원으로 flattening 합니다.
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])

# [0, 255] 사이의 값을 [0, 1]사이의 값으로 Normalize합니다.
x_train, x_test = x_train / 255., x_test / 255.

# 학습에 필요한 설정값들을 정의합니다.
learning_rate_RMSProp = 0.02
learning_rate_GradientDescent = 0.5
num_epochs = 100         # 반복횟수
batch_size = 256          
display_step = 1         # 몇 Step마다 log를 출력할지 결정합니다.
input_size = 784         # MNIST 데이터 input (이미지 크기: 28*28)
hidden1_size = 128       # 첫번째 히든레이어의 노드 개수 
hidden2_size = 64        # 두번째 히든레이어의 노드 개수 

# tf.data API를 이용해서 데이터를 섞고 batch 형태로 가져옵니다.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(60000).batch(batch_size)

# Autoencoder 모델을 정의합니다.
class AutoEncoder(object):
  # Autoencoder 모델을 위한 tf.Variable들을 정의합니다.
  def __init__(self):
    # 인코딩(Encoding) - 784 -> 128 -> 64
    self.Wh_1 = tf.Variable(tf.random.normal([input_size, hidden1_size]))
    self.bh_1 = tf.Variable(tf.random.normal([hidden1_size]))
    self.Wh_2 = tf.Variable(tf.random.normal([hidden1_size, hidden2_size]))
    self.bh_2 = tf.Variable(tf.random.normal([hidden2_size]))

    # 디코딩(Decoding) 64 -> 128 -> 784
    self.Wh_3 = tf.Variable(tf.random.normal([hidden2_size, hidden1_size]))
    self.bh_3 = tf.Variable(tf.random.normal([hidden1_size]))
    self.Wo = tf.Variable(tf.random.normal([hidden1_size, input_size]))
    self.bo = tf.Variable(tf.random.normal([input_size]))

  def __call__(self, x):
    H1_output = tf.nn.sigmoid(tf.matmul(x, self.Wh_1) + self.bh_1)
    H2_output = tf.nn.sigmoid(tf.matmul(H1_output, self.Wh_2) + self.bh_2)
    H3_output = tf.nn.sigmoid(tf.matmul(H2_output, self.Wh_3) + self.bh_3)
    X_reconstructed = tf.nn.sigmoid(tf.matmul(H3_output, self.Wo) + self.bo)

    return X_reconstructed, H2_output

# Softmax 분류기를 정의합니다.
class SoftmaxClassifier(object):
  # Softmax 모델을 위한 tf.Variable들을 정의합니다.
  def __init__(self):
    self.W_softmax = tf.Variable(tf.zeros([hidden2_size, 10]))  # 원본 MNIST 이미지(784) 대신 오토인코더의 압축된 특징(64)을 입력값으로 받습니다.
    self.b_softmax = tf.Variable(tf.zeros([10]))

  def __call__(self, x):
    y_pred = tf.nn.softmax(tf.matmul(x, self.W_softmax) + self.b_softmax)

    return y_pred


# MSE(Mean of Squared Error) 손실함수
@tf.function
def pretraining_mse_loss(y_pred, y_true):
  return tf.reduce_mean(tf.pow(y_true - y_pred, 2))


# cross-entropy loss 함수
@tf.function
def finetuning_cross_entropy_loss(y_pred_softmax, y):
  return tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(y_pred_softmax), axis=[1]))


# 1. Pre-Training : MNIST 데이터 재구축을 목적으로하는 옵티마이저와 최적화를 위한 function 정의합니다.
pretraining_optimizer = tf.optimizers.RMSprop(learning_rate_RMSProp, epsilon=1e-10)

@tf.function
def pretraining_train_step(autoencoder_model, x):
  # 타겟데이터는 인풋데이터와 같습니다.
  y_true = x
  with tf.GradientTape() as tape:
    y_pred, _ = autoencoder_model(x)
    pretraining_loss = pretraining_mse_loss(y_pred, y_true)
  gradients = tape.gradient(pretraining_loss, vars(autoencoder_model).values())
  pretraining_optimizer.apply_gradients(zip(gradients, vars(autoencoder_model).values()))


# 2. Fine-Tuning :  MNIST 데이터 분류를 목적으로하는 옵티마이저와 최적화를 위한 function 정의합니다.
finetuning_optimizer = tf.optimizers.SGD(learning_rate_GradientDescent)

@tf.function
def finetuning_train_step(autoencoder_model, softmax_classifier_model, x, y):
  with tf.GradientTape() as tape:
    y_pred, extracted_features = autoencoder_model(x)
    y_pred_softmax = softmax_classifier_model(extracted_features)
    finetuning_loss = finetuning_cross_entropy_loss(y_pred_softmax, y)
  autoencoder_encoding_variables = [autoencoder_model.Wh_1, autoencoder_model.bh_1, autoencoder_model.Wh_2, autoencoder_model.bh_2]
  gradients = tape.gradient(finetuning_loss, autoencoder_encoding_variables + list(vars(softmax_classifier_model).values()))
  finetuning_optimizer.apply_gradients(zip(gradients, autoencoder_encoding_variables + list(vars(softmax_classifier_model).values())))


# 모델의 정확도를 출력하는 함수를 정의합니다.
@tf.function
def compute_accuracy(y_pred_softmax, y):
  correct_prediction = tf.equal(tf.argmax(y_pred_softmax,1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return accuracy



# Autoencoder 모델을 선언합니다.
AutoEncoder_model = AutoEncoder()

# Softmax 분류기 모델을 선언합니다. (입력으로 Autoencoder의 압축된 특징을 넣습니다.)
SoftmaxClassifier_model = SoftmaxClassifier()

# Step 1: MNIST 데이터 재구축을 위한 오토인코더 최적화(Pre-Training)를 수행합니다.
for epoch in range(num_epochs):
  # 모든 배치들에 대해서 최적화를 수행합니다.
  for batch_x, _ in train_data:
    _, pretraining_loss_print = pretraining_train_step(AutoEncoder_model, batch_x), pretraining_mse_loss(AutoEncoder_model(batch_x)[0], batch_x)
  # 지정된 epoch마다 학습결과를 출력합니다.
  if epoch % display_step == 0:
    print("반복(Epoch): %d, Pre-Training 손실 함수(pretraining_loss): %f" % ((epoch + 1), pretraining_loss_print))
print("Step 1 : MNIST 데이터 재구축을 위한 오토인코더 최적화 완료(Pre-Training)")

# Step 2: MNIST 데이터 분류를 위한 오토인코더+Softmax 분류기 최적화(Fine-tuning)를 수행합니다.
for epoch in range(num_epochs + 100):
  # 모든 배치들에 대해서 최적화를 수행합니다.
  for batch_x, batch_y in train_data:
    batch_y = tf.one_hot(batch_y, depth=10)
    _, finetuning_loss_print = finetuning_train_step(AutoEncoder_model, SoftmaxClassifier_model, batch_x, batch_y), finetuning_cross_entropy_loss(SoftmaxClassifier_model(AutoEncoder_model(batch_x)[1]), batch_y)
  # 지정된 epoch마다 학습결과를 출력합니다.
  if epoch % display_step == 0:
    print("반복(Epoch): %d, Fine-tuning 손실 함수(finetuning_loss): %f" % ((epoch + 1), finetuning_loss_print))
print("Step 2 : MNIST 데이터 분류를 위한 오토인코더+Softmax 분류기 최적화 완료(Fine-Tuning)")

# 오토인코더+Softmax 분류기 모델의 정확도를 출력합니다.
print("정확도(오토인코더+Softmax 분류기): %f" % compute_accuracy(SoftmaxClassifier_model(AutoEncoder_model(x_test)[1]), tf.one_hot(y_test, depth=10)))  # 정확도 : 약 96%


'''
결과: 
반복(Epoch): 1, Pre-Training 손실 함수(pretraining_loss): 0.053507
반복(Epoch): 2, Pre-Training 손실 함수(pretraining_loss): 0.048763
반복(Epoch): 3, Pre-Training 손실 함수(pretraining_loss): 0.035184
반복(Epoch): 4, Pre-Training 손실 함수(pretraining_loss): 0.036432
반복(Epoch): 5, Pre-Training 손실 함수(pretraining_loss): 0.030125
반복(Epoch): 6, Pre-Training 손실 함수(pretraining_loss): 0.034440
반복(Epoch): 7, Pre-Training 손실 함수(pretraining_loss): 0.028495
반복(Epoch): 8, Pre-Training 손실 함수(pretraining_loss): 0.028204
반복(Epoch): 9, Pre-Training 손실 함수(pretraining_loss): 0.027567
반복(Epoch): 10, Pre-Training 손실 함수(pretraining_loss): 0.027792
반복(Epoch): 11, Pre-Training 손실 함수(pretraining_loss): 0.026380
반복(Epoch): 12, Pre-Training 손실 함수(pretraining_loss): 0.029198
반복(Epoch): 13, Pre-Training 손실 함수(pretraining_loss): 0.028753
반복(Epoch): 14, Pre-Training 손실 함수(pretraining_loss): 0.026498
반복(Epoch): 15, Pre-Training 손실 함수(pretraining_loss): 0.023731
반복(Epoch): 16, Pre-Training 손실 함수(pretraining_loss): 0.022695
반복(Epoch): 17, Pre-Training 손실 함수(pretraining_loss): 0.027665
반복(Epoch): 18, Pre-Training 손실 함수(pretraining_loss): 0.027225
반복(Epoch): 19, Pre-Training 손실 함수(pretraining_loss): 0.026682
반복(Epoch): 20, Pre-Training 손실 함수(pretraining_loss): 0.021913
반복(Epoch): 21, Pre-Training 손실 함수(pretraining_loss): 0.024350
반복(Epoch): 22, Pre-Training 손실 함수(pretraining_loss): 0.023734
반복(Epoch): 23, Pre-Training 손실 함수(pretraining_loss): 0.022152
반복(Epoch): 24, Pre-Training 손실 함수(pretraining_loss): 0.025833
반복(Epoch): 25, Pre-Training 손실 함수(pretraining_loss): 0.020808
반복(Epoch): 26, Pre-Training 손실 함수(pretraining_loss): 0.021201
반복(Epoch): 27, Pre-Training 손실 함수(pretraining_loss): 0.021918
반복(Epoch): 28, Pre-Training 손실 함수(pretraining_loss): 0.022194
반복(Epoch): 29, Pre-Training 손실 함수(pretraining_loss): 0.020438
반복(Epoch): 30, Pre-Training 손실 함수(pretraining_loss): 0.022512
반복(Epoch): 31, Pre-Training 손실 함수(pretraining_loss): 0.021349
반복(Epoch): 32, Pre-Training 손실 함수(pretraining_loss): 0.021471
반복(Epoch): 33, Pre-Training 손실 함수(pretraining_loss): 0.017022
반복(Epoch): 34, Pre-Training 손실 함수(pretraining_loss): 0.021992
반복(Epoch): 35, Pre-Training 손실 함수(pretraining_loss): 0.021022
반복(Epoch): 36, Pre-Training 손실 함수(pretraining_loss): 0.020439
반복(Epoch): 37, Pre-Training 손실 함수(pretraining_loss): 0.021232
반복(Epoch): 38, Pre-Training 손실 함수(pretraining_loss): 0.017217
반복(Epoch): 39, Pre-Training 손실 함수(pretraining_loss): 0.019105
반복(Epoch): 40, Pre-Training 손실 함수(pretraining_loss): 0.018256
반복(Epoch): 41, Pre-Training 손실 함수(pretraining_loss): 0.017459
반복(Epoch): 42, Pre-Training 손실 함수(pretraining_loss): 0.016606
반복(Epoch): 43, Pre-Training 손실 함수(pretraining_loss): 0.016887
반복(Epoch): 44, Pre-Training 손실 함수(pretraining_loss): 0.016546
반복(Epoch): 45, Pre-Training 손실 함수(pretraining_loss): 0.016624
반복(Epoch): 46, Pre-Training 손실 함수(pretraining_loss): 0.015461
반복(Epoch): 47, Pre-Training 손실 함수(pretraining_loss): 0.016206
반복(Epoch): 48, Pre-Training 손실 함수(pretraining_loss): 0.014506
반복(Epoch): 49, Pre-Training 손실 함수(pretraining_loss): 0.017969
반복(Epoch): 50, Pre-Training 손실 함수(pretraining_loss): 0.015498
반복(Epoch): 51, Pre-Training 손실 함수(pretraining_loss): 0.014712
반복(Epoch): 52, Pre-Training 손실 함수(pretraining_loss): 0.016367
반복(Epoch): 53, Pre-Training 손실 함수(pretraining_loss): 0.017104
반복(Epoch): 54, Pre-Training 손실 함수(pretraining_loss): 0.015905
반복(Epoch): 55, Pre-Training 손실 함수(pretraining_loss): 0.016148
반복(Epoch): 56, Pre-Training 손실 함수(pretraining_loss): 0.018872
반복(Epoch): 57, Pre-Training 손실 함수(pretraining_loss): 0.014031
반복(Epoch): 58, Pre-Training 손실 함수(pretraining_loss): 0.015222
반복(Epoch): 59, Pre-Training 손실 함수(pretraining_loss): 0.015184
반복(Epoch): 60, Pre-Training 손실 함수(pretraining_loss): 0.012931
반복(Epoch): 61, Pre-Training 손실 함수(pretraining_loss): 0.016481
반복(Epoch): 62, Pre-Training 손실 함수(pretraining_loss): 0.012996
반복(Epoch): 63, Pre-Training 손실 함수(pretraining_loss): 0.012893
반복(Epoch): 64, Pre-Training 손실 함수(pretraining_loss): 0.014411
반복(Epoch): 65, Pre-Training 손실 함수(pretraining_loss): 0.016494
반복(Epoch): 66, Pre-Training 손실 함수(pretraining_loss): 0.016149
반복(Epoch): 67, Pre-Training 손실 함수(pretraining_loss): 0.014902
반복(Epoch): 68, Pre-Training 손실 함수(pretraining_loss): 0.014104
반복(Epoch): 69, Pre-Training 손실 함수(pretraining_loss): 0.015446
반복(Epoch): 70, Pre-Training 손실 함수(pretraining_loss): 0.013116
반복(Epoch): 71, Pre-Training 손실 함수(pretraining_loss): 0.014651
반복(Epoch): 72, Pre-Training 손실 함수(pretraining_loss): 0.012222
반복(Epoch): 73, Pre-Training 손실 함수(pretraining_loss): 0.011534
반복(Epoch): 74, Pre-Training 손실 함수(pretraining_loss): 0.012763
반복(Epoch): 75, Pre-Training 손실 함수(pretraining_loss): 0.012654
반복(Epoch): 76, Pre-Training 손실 함수(pretraining_loss): 0.011128
반복(Epoch): 77, Pre-Training 손실 함수(pretraining_loss): 0.012303
반복(Epoch): 78, Pre-Training 손실 함수(pretraining_loss): 0.010430
반복(Epoch): 79, Pre-Training 손실 함수(pretraining_loss): 0.012946
반복(Epoch): 80, Pre-Training 손실 함수(pretraining_loss): 0.013003
반복(Epoch): 81, Pre-Training 손실 함수(pretraining_loss): 0.012357
반복(Epoch): 82, Pre-Training 손실 함수(pretraining_loss): 0.011040
반복(Epoch): 83, Pre-Training 손실 함수(pretraining_loss): 0.011767
반복(Epoch): 84, Pre-Training 손실 함수(pretraining_loss): 0.012648
반복(Epoch): 85, Pre-Training 손실 함수(pretraining_loss): 0.012419
반복(Epoch): 86, Pre-Training 손실 함수(pretraining_loss): 0.012792
반복(Epoch): 87, Pre-Training 손실 함수(pretraining_loss): 0.012415
반복(Epoch): 88, Pre-Training 손실 함수(pretraining_loss): 0.012451
반복(Epoch): 89, Pre-Training 손실 함수(pretraining_loss): 0.014441
반복(Epoch): 90, Pre-Training 손실 함수(pretraining_loss): 0.013401
반복(Epoch): 91, Pre-Training 손실 함수(pretraining_loss): 0.013060
반복(Epoch): 92, Pre-Training 손실 함수(pretraining_loss): 0.012423
반복(Epoch): 93, Pre-Training 손실 함수(pretraining_loss): 0.011332
반복(Epoch): 94, Pre-Training 손실 함수(pretraining_loss): 0.010706
반복(Epoch): 95, Pre-Training 손실 함수(pretraining_loss): 0.012547
반복(Epoch): 96, Pre-Training 손실 함수(pretraining_loss): 0.009669
반복(Epoch): 97, Pre-Training 손실 함수(pretraining_loss): 0.013293
반복(Epoch): 98, Pre-Training 손실 함수(pretraining_loss): 0.010423
반복(Epoch): 99, Pre-Training 손실 함수(pretraining_loss): 0.012039
반복(Epoch): 100, Pre-Training 손실 함수(pretraining_loss): 0.012134
Step 1 : MNIST 데이터 재구축을 위한 오토인코더 최적화 완료(Pre-Training)
반복(Epoch): 1, Fine-tuning 손실 함수(finetuning_loss): 0.325304
반복(Epoch): 2, Fine-tuning 손실 함수(finetuning_loss): 0.276397
반복(Epoch): 3, Fine-tuning 손실 함수(finetuning_loss): 0.382290
반복(Epoch): 4, Fine-tuning 손실 함수(finetuning_loss): 0.216175
반복(Epoch): 5, Fine-tuning 손실 함수(finetuning_loss): 0.318119
반복(Epoch): 6, Fine-tuning 손실 함수(finetuning_loss): 0.270805
반복(Epoch): 7, Fine-tuning 손실 함수(finetuning_loss): 0.240315
반복(Epoch): 8, Fine-tuning 손실 함수(finetuning_loss): 0.227046
반복(Epoch): 9, Fine-tuning 손실 함수(finetuning_loss): 0.367014
반복(Epoch): 10, Fine-tuning 손실 함수(finetuning_loss): 0.234238
반복(Epoch): 11, Fine-tuning 손실 함수(finetuning_loss): 0.214215
반복(Epoch): 12, Fine-tuning 손실 함수(finetuning_loss): 0.098432
반복(Epoch): 13, Fine-tuning 손실 함수(finetuning_loss): 0.129280
반복(Epoch): 14, Fine-tuning 손실 함수(finetuning_loss): 0.183790
반복(Epoch): 15, Fine-tuning 손실 함수(finetuning_loss): 0.121607
반복(Epoch): 16, Fine-tuning 손실 함수(finetuning_loss): 0.210740
반복(Epoch): 17, Fine-tuning 손실 함수(finetuning_loss): 0.193307
반복(Epoch): 18, Fine-tuning 손실 함수(finetuning_loss): 0.121441
반복(Epoch): 19, Fine-tuning 손실 함수(finetuning_loss): 0.094763
반복(Epoch): 20, Fine-tuning 손실 함수(finetuning_loss): 0.106741
반복(Epoch): 21, Fine-tuning 손실 함수(finetuning_loss): 0.146501
반복(Epoch): 22, Fine-tuning 손실 함수(finetuning_loss): 0.069261
반복(Epoch): 23, Fine-tuning 손실 함수(finetuning_loss): 0.093902
반복(Epoch): 24, Fine-tuning 손실 함수(finetuning_loss): 0.256915
반복(Epoch): 25, Fine-tuning 손실 함수(finetuning_loss): 0.189629
반복(Epoch): 26, Fine-tuning 손실 함수(finetuning_loss): 0.056574
반복(Epoch): 27, Fine-tuning 손실 함수(finetuning_loss): 0.071725
반복(Epoch): 28, Fine-tuning 손실 함수(finetuning_loss): 0.135041
반복(Epoch): 29, Fine-tuning 손실 함수(finetuning_loss): 0.147878
반복(Epoch): 30, Fine-tuning 손실 함수(finetuning_loss): 0.107991
반복(Epoch): 31, Fine-tuning 손실 함수(finetuning_loss): 0.123838
반복(Epoch): 32, Fine-tuning 손실 함수(finetuning_loss): 0.111915
반복(Epoch): 33, Fine-tuning 손실 함수(finetuning_loss): 0.040454
반복(Epoch): 34, Fine-tuning 손실 함수(finetuning_loss): 0.096574
반복(Epoch): 35, Fine-tuning 손실 함수(finetuning_loss): 0.029988
반복(Epoch): 36, Fine-tuning 손실 함수(finetuning_loss): 0.063732
반복(Epoch): 37, Fine-tuning 손실 함수(finetuning_loss): 0.108929
반복(Epoch): 38, Fine-tuning 손실 함수(finetuning_loss): 0.064120
반복(Epoch): 39, Fine-tuning 손실 함수(finetuning_loss): 0.099476
반복(Epoch): 40, Fine-tuning 손실 함수(finetuning_loss): 0.073152
반복(Epoch): 41, Fine-tuning 손실 함수(finetuning_loss): 0.060018
반복(Epoch): 42, Fine-tuning 손실 함수(finetuning_loss): 0.183250
반복(Epoch): 43, Fine-tuning 손실 함수(finetuning_loss): 0.058556
반복(Epoch): 44, Fine-tuning 손실 함수(finetuning_loss): 0.070945
반복(Epoch): 45, Fine-tuning 손실 함수(finetuning_loss): 0.040676
반복(Epoch): 46, Fine-tuning 손실 함수(finetuning_loss): 0.043457
반복(Epoch): 47, Fine-tuning 손실 함수(finetuning_loss): 0.078201
반복(Epoch): 48, Fine-tuning 손실 함수(finetuning_loss): 0.052978
반복(Epoch): 49, Fine-tuning 손실 함수(finetuning_loss): 0.043417
반복(Epoch): 50, Fine-tuning 손실 함수(finetuning_loss): 0.034806
반복(Epoch): 51, Fine-tuning 손실 함수(finetuning_loss): 0.030281
반복(Epoch): 52, Fine-tuning 손실 함수(finetuning_loss): 0.049490
반복(Epoch): 53, Fine-tuning 손실 함수(finetuning_loss): 0.043214
반복(Epoch): 54, Fine-tuning 손실 함수(finetuning_loss): 0.056933
반복(Epoch): 55, Fine-tuning 손실 함수(finetuning_loss): 0.045052
반복(Epoch): 56, Fine-tuning 손실 함수(finetuning_loss): 0.115746
반복(Epoch): 57, Fine-tuning 손실 함수(finetuning_loss): 0.032351
반복(Epoch): 58, Fine-tuning 손실 함수(finetuning_loss): 0.032108
반복(Epoch): 59, Fine-tuning 손실 함수(finetuning_loss): 0.053930
반복(Epoch): 60, Fine-tuning 손실 함수(finetuning_loss): 0.024719
반복(Epoch): 61, Fine-tuning 손실 함수(finetuning_loss): 0.024479
반복(Epoch): 62, Fine-tuning 손실 함수(finetuning_loss): 0.055692
반복(Epoch): 63, Fine-tuning 손실 함수(finetuning_loss): 0.056284
반복(Epoch): 64, Fine-tuning 손실 함수(finetuning_loss): 0.067199
반복(Epoch): 65, Fine-tuning 손실 함수(finetuning_loss): 0.038257
반복(Epoch): 66, Fine-tuning 손실 함수(finetuning_loss): 0.051612
반복(Epoch): 67, Fine-tuning 손실 함수(finetuning_loss): 0.063251
반복(Epoch): 68, Fine-tuning 손실 함수(finetuning_loss): 0.035636
반복(Epoch): 69, Fine-tuning 손실 함수(finetuning_loss): 0.039104
반복(Epoch): 70, Fine-tuning 손실 함수(finetuning_loss): 0.083766
반복(Epoch): 71, Fine-tuning 손실 함수(finetuning_loss): 0.057188
반복(Epoch): 72, Fine-tuning 손실 함수(finetuning_loss): 0.044378
반복(Epoch): 73, Fine-tuning 손실 함수(finetuning_loss): 0.063846
반복(Epoch): 74, Fine-tuning 손실 함수(finetuning_loss): 0.023192
반복(Epoch): 75, Fine-tuning 손실 함수(finetuning_loss): 0.022278
반복(Epoch): 76, Fine-tuning 손실 함수(finetuning_loss): 0.023742
반복(Epoch): 77, Fine-tuning 손실 함수(finetuning_loss): 0.030204
반복(Epoch): 78, Fine-tuning 손실 함수(finetuning_loss): 0.061614
반복(Epoch): 79, Fine-tuning 손실 함수(finetuning_loss): 0.027759
반복(Epoch): 80, Fine-tuning 손실 함수(finetuning_loss): 0.043625
반복(Epoch): 81, Fine-tuning 손실 함수(finetuning_loss): 0.037367
반복(Epoch): 82, Fine-tuning 손실 함수(finetuning_loss): 0.045685
반복(Epoch): 83, Fine-tuning 손실 함수(finetuning_loss): 0.058006
반복(Epoch): 84, Fine-tuning 손실 함수(finetuning_loss): 0.043170
반복(Epoch): 85, Fine-tuning 손실 함수(finetuning_loss): 0.037628
반복(Epoch): 86, Fine-tuning 손실 함수(finetuning_loss): 0.018788
반복(Epoch): 87, Fine-tuning 손실 함수(finetuning_loss): 0.028692
반복(Epoch): 88, Fine-tuning 손실 함수(finetuning_loss): 0.098316
반복(Epoch): 89, Fine-tuning 손실 함수(finetuning_loss): 0.053149
반복(Epoch): 90, Fine-tuning 손실 함수(finetuning_loss): 0.037437
반복(Epoch): 91, Fine-tuning 손실 함수(finetuning_loss): 0.014776
반복(Epoch): 92, Fine-tuning 손실 함수(finetuning_loss): 0.040170
반복(Epoch): 93, Fine-tuning 손실 함수(finetuning_loss): 0.036712
반복(Epoch): 94, Fine-tuning 손실 함수(finetuning_loss): 0.027032
반복(Epoch): 95, Fine-tuning 손실 함수(finetuning_loss): 0.050907
반복(Epoch): 96, Fine-tuning 손실 함수(finetuning_loss): 0.031556
반복(Epoch): 97, Fine-tuning 손실 함수(finetuning_loss): 0.019665
반복(Epoch): 98, Fine-tuning 손실 함수(finetuning_loss): 0.031778
반복(Epoch): 99, Fine-tuning 손실 함수(finetuning_loss): 0.042951
반복(Epoch): 100, Fine-tuning 손실 함수(finetuning_loss): 0.030048
반복(Epoch): 101, Fine-tuning 손실 함수(finetuning_loss): 0.009764
반복(Epoch): 102, Fine-tuning 손실 함수(finetuning_loss): 0.036724
반복(Epoch): 103, Fine-tuning 손실 함수(finetuning_loss): 0.021788
반복(Epoch): 104, Fine-tuning 손실 함수(finetuning_loss): 0.028175
반복(Epoch): 105, Fine-tuning 손실 함수(finetuning_loss): 0.057793
반복(Epoch): 106, Fine-tuning 손실 함수(finetuning_loss): 0.009467
반복(Epoch): 107, Fine-tuning 손실 함수(finetuning_loss): 0.010038
반복(Epoch): 108, Fine-tuning 손실 함수(finetuning_loss): 0.015010
반복(Epoch): 109, Fine-tuning 손실 함수(finetuning_loss): 0.049014
반복(Epoch): 110, Fine-tuning 손실 함수(finetuning_loss): 0.019715
반복(Epoch): 111, Fine-tuning 손실 함수(finetuning_loss): 0.036950
반복(Epoch): 112, Fine-tuning 손실 함수(finetuning_loss): 0.023561
반복(Epoch): 113, Fine-tuning 손실 함수(finetuning_loss): 0.050550
반복(Epoch): 114, Fine-tuning 손실 함수(finetuning_loss): 0.021130
반복(Epoch): 115, Fine-tuning 손실 함수(finetuning_loss): 0.047228
반복(Epoch): 116, Fine-tuning 손실 함수(finetuning_loss): 0.023643
반복(Epoch): 117, Fine-tuning 손실 함수(finetuning_loss): 0.027961
반복(Epoch): 118, Fine-tuning 손실 함수(finetuning_loss): 0.014484
반복(Epoch): 119, Fine-tuning 손실 함수(finetuning_loss): 0.014494
반복(Epoch): 120, Fine-tuning 손실 함수(finetuning_loss): 0.016496
반복(Epoch): 121, Fine-tuning 손실 함수(finetuning_loss): 0.062865
반복(Epoch): 122, Fine-tuning 손실 함수(finetuning_loss): 0.054801
반복(Epoch): 123, Fine-tuning 손실 함수(finetuning_loss): 0.031489
반복(Epoch): 124, Fine-tuning 손실 함수(finetuning_loss): 0.010628
반복(Epoch): 125, Fine-tuning 손실 함수(finetuning_loss): 0.023272
반복(Epoch): 126, Fine-tuning 손실 함수(finetuning_loss): 0.004501
반복(Epoch): 127, Fine-tuning 손실 함수(finetuning_loss): 0.022569
반복(Epoch): 128, Fine-tuning 손실 함수(finetuning_loss): 0.020074
반복(Epoch): 129, Fine-tuning 손실 함수(finetuning_loss): 0.022864
반복(Epoch): 130, Fine-tuning 손실 함수(finetuning_loss): 0.013387
반복(Epoch): 131, Fine-tuning 손실 함수(finetuning_loss): 0.014080
반복(Epoch): 132, Fine-tuning 손실 함수(finetuning_loss): 0.029491
반복(Epoch): 133, Fine-tuning 손실 함수(finetuning_loss): 0.027683
반복(Epoch): 134, Fine-tuning 손실 함수(finetuning_loss): 0.009023
반복(Epoch): 135, Fine-tuning 손실 함수(finetuning_loss): 0.024363
반복(Epoch): 136, Fine-tuning 손실 함수(finetuning_loss): 0.021771
반복(Epoch): 137, Fine-tuning 손실 함수(finetuning_loss): 0.018908
반복(Epoch): 138, Fine-tuning 손실 함수(finetuning_loss): 0.010672
반복(Epoch): 139, Fine-tuning 손실 함수(finetuning_loss): 0.008865
반복(Epoch): 140, Fine-tuning 손실 함수(finetuning_loss): 0.016291
반복(Epoch): 141, Fine-tuning 손실 함수(finetuning_loss): 0.020377
반복(Epoch): 142, Fine-tuning 손실 함수(finetuning_loss): 0.021067
반복(Epoch): 143, Fine-tuning 손실 함수(finetuning_loss): 0.018236
반복(Epoch): 144, Fine-tuning 손실 함수(finetuning_loss): 0.024964
반복(Epoch): 145, Fine-tuning 손실 함수(finetuning_loss): 0.014431
반복(Epoch): 146, Fine-tuning 손실 함수(finetuning_loss): 0.023570
반복(Epoch): 147, Fine-tuning 손실 함수(finetuning_loss): 0.012575
반복(Epoch): 148, Fine-tuning 손실 함수(finetuning_loss): 0.011586
반복(Epoch): 149, Fine-tuning 손실 함수(finetuning_loss): 0.029435
반복(Epoch): 150, Fine-tuning 손실 함수(finetuning_loss): 0.010515
반복(Epoch): 151, Fine-tuning 손실 함수(finetuning_loss): 0.019928
반복(Epoch): 152, Fine-tuning 손실 함수(finetuning_loss): 0.023447
반복(Epoch): 153, Fine-tuning 손실 함수(finetuning_loss): 0.017635
반복(Epoch): 154, Fine-tuning 손실 함수(finetuning_loss): 0.014962
반복(Epoch): 155, Fine-tuning 손실 함수(finetuning_loss): 0.027726
반복(Epoch): 156, Fine-tuning 손실 함수(finetuning_loss): 0.019980
반복(Epoch): 157, Fine-tuning 손실 함수(finetuning_loss): 0.008557
반복(Epoch): 158, Fine-tuning 손실 함수(finetuning_loss): 0.011651
반복(Epoch): 159, Fine-tuning 손실 함수(finetuning_loss): 0.007926
반복(Epoch): 160, Fine-tuning 손실 함수(finetuning_loss): 0.015225
반복(Epoch): 161, Fine-tuning 손실 함수(finetuning_loss): 0.012383
반복(Epoch): 162, Fine-tuning 손실 함수(finetuning_loss): 0.013008
반복(Epoch): 163, Fine-tuning 손실 함수(finetuning_loss): 0.017309
반복(Epoch): 164, Fine-tuning 손실 함수(finetuning_loss): 0.034151
반복(Epoch): 165, Fine-tuning 손실 함수(finetuning_loss): 0.013219
반복(Epoch): 166, Fine-tuning 손실 함수(finetuning_loss): 0.021021
반복(Epoch): 167, Fine-tuning 손실 함수(finetuning_loss): 0.008926
반복(Epoch): 168, Fine-tuning 손실 함수(finetuning_loss): 0.035952
반복(Epoch): 169, Fine-tuning 손실 함수(finetuning_loss): 0.008305
반복(Epoch): 170, Fine-tuning 손실 함수(finetuning_loss): 0.010765
반복(Epoch): 171, Fine-tuning 손실 함수(finetuning_loss): 0.013321
반복(Epoch): 172, Fine-tuning 손실 함수(finetuning_loss): 0.019369
반복(Epoch): 173, Fine-tuning 손실 함수(finetuning_loss): 0.033441
반복(Epoch): 174, Fine-tuning 손실 함수(finetuning_loss): 0.012628
반복(Epoch): 175, Fine-tuning 손실 함수(finetuning_loss): 0.022365
반복(Epoch): 176, Fine-tuning 손실 함수(finetuning_loss): 0.006675
반복(Epoch): 177, Fine-tuning 손실 함수(finetuning_loss): 0.011718
반복(Epoch): 178, Fine-tuning 손실 함수(finetuning_loss): 0.012873
반복(Epoch): 179, Fine-tuning 손실 함수(finetuning_loss): 0.012914
반복(Epoch): 180, Fine-tuning 손실 함수(finetuning_loss): 0.005970
반복(Epoch): 181, Fine-tuning 손실 함수(finetuning_loss): 0.017133
반복(Epoch): 182, Fine-tuning 손실 함수(finetuning_loss): 0.013680
반복(Epoch): 183, Fine-tuning 손실 함수(finetuning_loss): 0.020669
반복(Epoch): 184, Fine-tuning 손실 함수(finetuning_loss): 0.008793
반복(Epoch): 185, Fine-tuning 손실 함수(finetuning_loss): 0.023708
반복(Epoch): 186, Fine-tuning 손실 함수(finetuning_loss): 0.007231
반복(Epoch): 187, Fine-tuning 손실 함수(finetuning_loss): 0.011926
반복(Epoch): 188, Fine-tuning 손실 함수(finetuning_loss): 0.007000
반복(Epoch): 189, Fine-tuning 손실 함수(finetuning_loss): 0.012216
반복(Epoch): 190, Fine-tuning 손실 함수(finetuning_loss): 0.011874
반복(Epoch): 191, Fine-tuning 손실 함수(finetuning_loss): 0.011675
반복(Epoch): 192, Fine-tuning 손실 함수(finetuning_loss): 0.014625
반복(Epoch): 193, Fine-tuning 손실 함수(finetuning_loss): 0.014802
반복(Epoch): 194, Fine-tuning 손실 함수(finetuning_loss): 0.005100
반복(Epoch): 195, Fine-tuning 손실 함수(finetuning_loss): 0.011093
반복(Epoch): 196, Fine-tuning 손실 함수(finetuning_loss): 0.016482
반복(Epoch): 197, Fine-tuning 손실 함수(finetuning_loss): 0.006218
반복(Epoch): 198, Fine-tuning 손실 함수(finetuning_loss): 0.022026
반복(Epoch): 199, Fine-tuning 손실 함수(finetuning_loss): 0.019975
반복(Epoch): 200, Fine-tuning 손실 함수(finetuning_loss): 0.009672
Step 2 : MNIST 데이터 분류를 위한 오토인코더+Softmax 분류기 최적화 완료(Fine-Tuning)
정확도(오토인코더+Softmax 분류기): 0.962000
'''