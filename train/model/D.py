import tensorflow as tf
import tensorflow_addons as tfa
from tools import ReflectionPadding2D


class Block(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Block, self).__init__()
        self.conv=tf.keras.Sequential([
            ReflectionPadding2D(),
            tf.keras.layers.Conv2D(padding='valid',**kwargs),
            tfa.layers.InstanceNormalization(axis=-1),
            tf.keras.layers.LeakyReLU(alpha=0.2)
        ])

    def call(self,x,training=False):
        return self.conv(x,training=training)

'''
a=Block(filters=3,kernel_size=3,strides=1)
x=tf.random.normal((1,202,202,3))
print(a(x).shape)
'''

class Discriminator(tf.keras.Model):
    def __init__(self,filters=[64,128,256,512]):
        super(Discriminator, self).__init__()
        self.init=tf.keras.Sequential([
            ReflectionPadding2D(),
            tf.keras.layers.Conv2D(filters=filters[0],kernel_size=4,strides=2,padding='valid'),
            tf.keras.layers.LeakyReLU(alpha=0.2)
        ])
#(256-4+2)
        self.layer=tf.keras.Sequential()
        for filter in filters[1:]:
            self.layer.add(Block(filters=filter,kernel_size=4,strides=1 if filter==filters[-1] else 2))
        self.layer.add(ReflectionPadding2D())
        self.layer.add(tf.keras.layers.Conv2D(filters=1,kernel_size=4,strides=1,padding='valid'))


    def call(self, x, training=False):
        x=self.init(x,training=training)
        x=self.layer(x,training=training)
        return tf.nn.sigmoid(x)

'''
D=Discriminator()
x=tf.random.normal((1,256,256,3))
print(D(x).shape)
print(D.summary())
'''





