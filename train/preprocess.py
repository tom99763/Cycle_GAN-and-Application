import tensorflow as tf


def get_data(pathx,pathy,batch_size,image_size=(256,256)):
    ds_X=tf.keras.preprocessing.image_dataset_from_directory(
        directory=pathx,
        shuffle=False,
        image_size=image_size,
        batch_size=1
    )

    ds_Y=tf.keras.preprocessing.image_dataset_from_directory(
        directory=pathy,
        shuffle=False,
        image_size=image_size,
        batch_size=1
    )

    ds=tf.data.Dataset.zip((ds_X,ds_Y)).shuffle(1000).batch(batch_size,drop_remainder=False).\
        cache().prefetch(tf.data.experimental.AUTOTUNE)

    return ds

'''
ds=get_data('../data/base_data/trainX','../data/base_data/trainY',32)
for x,y in ds:
    print(x[0][:,0,...].shape,y[0][:,0,...].shape)
'''