import tensorflow as tf
from utils import ReflectionPadding2D
import tensorflow_addons as tfa

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self,downsample=True,active=True,**kwargs):
        super(ConvBlock, self).__init__()
        self.conv=tf.keras.Sequential([
            ReflectionPadding2D() if downsample\
                else tf.keras.layers.Lambda(lambda x:x),
            tf.keras.layers.Conv2D(padding='valid',**kwargs) if downsample\
                else tf.keras.layers.Conv2DTranspose(**kwargs),
            tfa.layers.InstanceNormalization(axis=-1),
            tf.keras.layers.ReLU() if active \
                else tf.keras.layers.Lambda(lambda x:x)
        ])
    def call(self,x,training=False):
        return self.conv(x,training=training)

# Residual Dense Block
class ResDesBlock(tf.keras.layers.Layer):
    def __init__(self,filters):
        super(ResDesBlock,self).__init__()
        self.block1=ConvBlock(filters=filters,kernel_size=3,strides=1)
        self.block2=ConvBlock(filters=filters,kernel_size=3,active=False,strides=1)
        self.concat=tf.keras.layers.Concatenate(axis=-1)

    def call(self,x,training=False):
        o1=self.block1(x,training=training)
        o1=self.concat([x,o1])
        o2=self.block2(o1,training=training)
        output=x+o2
        return tf.nn.relu(output)
'''
RDB=ResDesBlock(128)
x=tf.random.normal((1,32,32,128))
print(RDB(x).shape)
'''


class Generator(tf.keras.Model):
    def __init__(self,filters=64,num_rdb=9):
        super(Generator,self).__init__()

        self.down=tf.keras.Sequential([
            ReflectionPadding2D((3,3)),
            tf.keras.layers.Conv2D(filters=filters, kernel_size=7, strides=1, activation='relu'),
            ConvBlock(filters=filters*2,kernel_size=3,strides=2),
            ConvBlock(filters=filters*4,kernel_size=3,strides=2),
        ])

        self.rdb_extractor=tf.keras.Sequential([ResDesBlock(filters*4) for _ in range(num_rdb)])

        self.up=tf.keras.Sequential([
            ConvBlock(filters=filters*2,kernel_size=3,strides=2,padding='same',output_padding=1,downsample=False),
            ConvBlock(filters=filters,kernel_size=3,strides=2,padding='same',output_padding=1,downsample=False),
            ReflectionPadding2D((3,3)),
            tf.keras.layers.Conv2D(filters=3, kernel_size=7, strides=1)
        ])

    def call(self,x,training=False):
        x=self.down(x,training=training)
        x=self.rdb_extractor(x,training=training)
        x=self.up(x,training=training)
        return tf.nn.tanh(x)
'''
G=Generator()
x=tf.random.normal((1,128,128,3))
print(G(x).shape)
'''