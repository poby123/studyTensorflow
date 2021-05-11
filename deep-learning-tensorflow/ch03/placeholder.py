import tensorflow as tf

# 2개의 값을 더하는 함수를 정의
@tf.function
def add_two_values(x, y):
    return x + y

# 출력: 7.5
print(add_two_values(3, 4.5).numpy())

# =======================
# 조금 더 복잡한 그래프 형태
@tf.function
def add_two_values_and_muliply_three(x, y):
    return 3 * add_two_values(x, y)

# 출력: 22.5
print(add_two_values_and_muliply_three(3, 4.5).numpy())