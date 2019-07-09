import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
grafik=[]
#din,fen,mat,sosyal,Türkçe,İngilizce                                                                       #din,fen,mat,sosyal,Türkçe,İngilizce
x=np.array([96.6677,93.75,86.875,95.,86.25,93.3750,96.6667,92.50,86.60,93.3333,90.,85.9375,89.25,92.,97.5,96.,88.8,96.8,95.,94.2,94.2857,100.,98.6,97.,98.25,93,99.2857,100.,97.2,96.])/100#kaynak not
s_k=np.array([5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,7.,7.,7.,7.,7.,7.])#sınıf kaynak
d_k=np.array([1.,1.,1.,1.,1.,1.,2.,2.,2.,2.,2.,2.,1.,1.,1.,1.,1.,1.,2.,2.,2.,2.,2.,2.,1.,1.,1.,1.,1.,1.])#dönem kaynak
s_h=np.array([5.,5.,5.,5.,5.,5.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,7.,7.,7.,7.,7.,7.,7.,7.,7.,7.,7.,7.])#sınıf hedef
d_h=np.array([2.,2.,2.,2.,2.,2.,1.,1.,1.,1.,1.,1.,2.,2.,2.,2.,2.,2.,1.,1.,1.,1.,1.,1.,2.,2.,2.,2.,2.,2.])#dönem hedef
y=np.array([96.6667,92.50,86.60,93.3333,90.,85.9375,89.25,92.,97.5,96.,88.8,96.8,95.,94.2,94.2857,100.,98.6,97.,98.25,93,99.2857,100.,97.2,96.,98.75,90.,100,98.,95.,96.])#hedef not
y=(y/100).transpose().astype(np.float32)
dataset=np.array([x,s_k,d_k,s_h,d_h]).transpose().astype(np.float32)
del x,s_k,d_k,s_h,d_h

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)


W0=tf.Variable(tf.zeros([5,3]))
b0=tf.Variable(tf.zeros([3]))

W1=tf.Variable(tf.zeros([3,2]))
b1=tf.Variable(tf.zeros([2]))

W2=tf.Variable(tf.zeros([2,1]))
b2=tf.Variable(tf.zeros([1]))

y0=tf.nn.tanh(tf.matmul(X,W0)+b0)
y1=tf.nn.relu(tf.matmul(y0,W1)+b1)
y_pred=tf.nn.tanh(tf.matmul(y1,W2)+b2)

loss=tf.reduce_sum(tf.square(Y-y_pred))

optimizer=tf.train.AdamOptimizer(1e-3).minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(100001):
    sess.run(optimizer,feed_dict={X:dataset,Y:y})
    if i%100==0:
      l=sess.run(loss,feed_dict={X:dataset,Y:y})
      print("Iteration",i,"Loss="+str(l))
  print(sess.run(y_pred,feed_dict={X:np.array([[0.96,7.,2.,8.,1.],[1.,1.,1.,1.,1.]]).astype(np.float32)}))
