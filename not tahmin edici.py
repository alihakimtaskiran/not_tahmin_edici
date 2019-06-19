import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
grafik=[]
#din,fen,mat,sosyal,Türkçe,İngilizce                                                                       #din,fen,mat,sosyal,Türkçe,İngilizce
x=np.array([96.6677,93.75,86.875,95.,86.25,93.3750,96.6667,92.50,86.60,93.3333,90.,85.9375,89.25,92.,97.5,96.,88.8,96.8,95.,94.2,94.2857,100.,98.6,97.,98.25,93,99.2857,100.,97.2,96.])#kaynak not
s_k=np.array([5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,7.,7.,7.,7.,7.,7.])#sınıf kaynak
d_k=np.array([1.,1.,1.,1.,1.,1.,2.,2.,2.,2.,2.,2.,1.,1.,1.,1.,1.,1.,2.,2.,2.,2.,2.,2.,1.,1.,1.,1.,1.,1.])#dönem kaynak
s_h=np.array([5.,5.,5.,5.,5.,5.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,6.,7.,7.,7.,7.,7.,7.,7.,7.,7.,7.,7.,7.])#sınıf hedef
d_h=np.array([2.,2.,2.,2.,2.,2.,1.,1.,1.,1.,1.,1.,2.,2.,2.,2.,2.,2.,1.,1.,1.,1.,1.,1.,2.,2.,2.,2.,2.,2.])#dönem hedef
y=np.array([96.6667,92.50,86.60,93.3333,90.,85.9375,89.25,92.,97.5,96.,88.8,96.8,95.,94.2,94.2857,100.,98.6,97.,98.25,93,99.2857,100.,97.2,96.,98.75,90.,100,98.,95.,96.])#hedef not

dataset=np.array([x,s_k,d_k,s_h,d_h,y])
del x,s_k,d_k,s_h,d_h,y
X0=tf.placeholder("float")
X1=tf.placeholder("float")
X2=tf.placeholder("float")
X3=tf.placeholder("float")
X4=tf.placeholder("float")
Y=tf.placeholder("float")

#define parameters

W0_0=tf.Variable(0.,name="W0_0")
W0_1=tf.Variable(0.,name="W0_1")
W0_2=tf.Variable(0.,name="W0_2")
W0_3=tf.Variable(0.,name="W0_3")
W0_4=tf.Variable(0.,name="W0_4")
b=tf.Variable(np.random.rand(),name="b")

toplam=tf.multiply(W0_0,X0)+tf.multiply(W0_1,X1)+tf.multiply(W0_2,X2)+tf.multiply(W0_3,X3)+tf.multiply(W0_4,X4)+b
y_tahmin=tf.nn.sigmoid(toplam)*100.

loss=tf.reduce_sum(tf.abs(Y-y_tahmin))#L1
optimizer=tf.train.AdamOptimizer(2e-4).minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(100000):
    sess.run(optimizer,feed_dict={X0:dataset[0],X1:dataset[1],X2:dataset[2],X3:dataset[3],X4:dataset[4],Y:dataset[5]})
    if i%100==0:
      l_=sess.run(loss,feed_dict={X0:dataset[0],X1:dataset[1],X2:dataset[2],X3:dataset[3],X4:dataset[4],Y:dataset[5]})
      grafik.append(l_)
      print("Döngü",i,"Loss="+str(l_))
  print(sess.run(y_tahmin,feed_dict={X0:96.,X1:7.,X2:2.,X3:8.,X4:1.}))
  sess.close()

plt.title("Loss Grafiği")
plt.plot(grafik,"-g.")
plt.show()

plt.title("Notlar")
plt.plot(dataset[0],"-.g")
plt.plot(dataset[5],"-.r")
plt.show()