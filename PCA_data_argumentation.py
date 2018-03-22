import tensorflow as tf
import cv2
import numpy as np

img = cv2.imread("7.jpg")
r,c,d = img.shape

x = np.array([[0,0,0] for _ in range(r*c)])

k = 0
for i in range(r):
    for j in range(c):
        x[k] = img[i][j]
        k += 1


#A = tf.Variable(x, dtype=tf.float64, name="A", expected_shape=(2,4))
B = tf.Variable(np.cov(x.transpose()))
C, D = tf.linalg.eigh(B, name="C")

E = tf.matmul(D, tf.reshape(C, (3,1))*np.random.normal(0,0.01), name="E")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

L = sess.run(tf.transpose(E))
x = x + L
print(sess.run(C))
k = 0
for i in range(r):
    for j in range(c):
        img[i][j] = x[k]
        k += 1


cv2.imshow("a", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
sess.close()
