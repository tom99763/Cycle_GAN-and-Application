import tensorflow as tf
from config import Config
from model.D import Discriminator
from model.G import Generator
from preprocess import get_data
from tqdm import tqdm
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

aug = tf.keras.Sequential([
    #tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    #tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, ),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5,offset=-1)
])

params=Config()


def flow(x,y,l1,l2,Dx,Dy,Gxy,Gyx,Dx_opt,Dy_opt,Gxy_opt,Gyx_opt):
    with tf.GradientTape(persistent=True) as tape:
        #cycle
        fake_y = Gxy(x, training=True) #x-->y'
        cycled_x = Gyx(fake_y, training=True) #y'-->x'
        fake_x = Gyx(y, training=True) #y-->x'
        cycled_y = Gxy(fake_x, training=True) #x'-->y'
        real_x=Gyx(x,training=True)
        real_y=Gxy(y,training=True)

        #Dloss
        Dx_loss=l2(Dx(x,training=True),tf.ones_like(Dx(x,training=True)))+\
            l2(Dx(fake_x,training=True),tf.zeros_like(Dx(fake_x,training=True)))

        Dx_loss*=params.lambda_dis

        Dy_loss = l2(Dy(y, training=True), tf.ones_like(Dy(y, training=True))) + \
                 l2(Dy(fake_y, training=True), tf.zeros_like(Dy(fake_y, training=True)))
        Dy_loss*=params.lambda_dis

        #Gloss
        Gxy_loss=l2(Dy(fake_y,training=True),tf.ones_like(Dy(fake_y,training=True)))
        Gyx_loss=l2(Dx(fake_x,training=True),tf.ones_like(Dx(fake_x,training=True)))

        #cycledLoss
        cycled_x_loss=l1(x,cycled_x)
        cycled_y_loss=l1(y,cycled_y)
        Cycled_loss=cycled_x_loss+cycled_y_loss

        #identityloss
        identity_x_loss=l1(x,real_x)
        identity_y_loss=l1(y,real_y)

        #total G
        total_Gxy_loss=Gxy_loss+\
                       params.lambda_cycle*Cycled_loss+\
                       params.lambda_identity*identity_y_loss #x-->y
        total_Gyx_loss=Gyx_loss+\
                       params.lambda_cycle*Cycled_loss+\
                       params.lambda_identity*identity_x_loss #y-->x

    Gxy_grads = tape.gradient(total_Gxy_loss,Gxy.trainable_variables)
    Gyx_grads = tape.gradient(total_Gyx_loss,Gyx.trainable_variables)
    Dx_grads = tape.gradient(Dx_loss,Dx.trainable_variables)
    Dy_grads = tape.gradient(Dy_loss, Dy.trainable_variables)

    Gxy_opt.apply_gradients(zip(Gxy_grads,Gxy.trainable_variables))
    Gyx_opt.apply_gradients(zip(Gyx_grads, Gyx.trainable_variables))
    Dx_opt.apply_gradients(zip(Dx_grads, Dx.trainable_variables))
    Dy_opt.apply_gradients(zip(Dy_grads, Dy.trainable_variables))
    return total_Gxy_loss,total_Gyx_loss,Dx_loss,Dy_loss


def test(ds,Dx,Dy,Gxy,Gyx):
    for i,(x,y) in enumerate(ds):
        if i==5:
            return
        x, y = x[0][:, 0, ...], y[0][:, 0, ...]
        x, y = aug(x), aug(y)

        y_hat=Gxy(x,training=False)[0,...]*127.5+127.5
        x_hat=Gyx(y,training=False)[0,...]*127.5+127.5

        y_hat=y_hat.numpy().astype('uint8')
        x_hat=x_hat.numpy().astype('uint8')

        cv2.imwrite(f'./generated_img/X2Y_{i}.jpg',y_hat)
        cv2.imwrite(f'./generated_img/Y2X_{i}.jpg',x_hat)

def main():
    #D out 30x30x1 sigmoid
    Dx=Discriminator()
    Dy=Discriminator()
    #G out same size from origin image
    Gxy=Generator()
    Gyx=Generator()

    Dx_optim=tf.keras.optimizers.Adam(learning_rate=params.lr,beta_1=0.5,beta_2=0.999)
    Dy_optim=tf.keras.optimizers.Adam(learning_rate=params.lr, beta_1=0.5, beta_2=0.999)
    Gxy_optim=tf.keras.optimizers.Adam(learning_rate=params.lr,beta_1=0.5,beta_2=0.999)
    Gyx_optim =tf.keras.optimizers.Adam(learning_rate=params.lr, beta_1=0.5, beta_2=0.999)

    ckpt = tf.train.Checkpoint(Dx=Dx, Dy=Dy, Gxy=Gxy, Gyx=Gyx, Dx_optim=Dx_optim, Dy_optim=Dy_optim, Gxy_optim=Gxy_optim, Gyx_optim=Gyx_optim)
    ckpt_manager = tf.train.CheckpointManager(ckpt, params.ckpt_path, max_to_keep=1)
    if ckpt_manager.latest_checkpoint and params.load_model:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('---ckpt restored----')

    l1=tf.keras.losses.MeanAbsoluteError()
    l2=tf.keras.losses.MeanSquaredError()

    ds_train=get_data(params.train_X_path,params.train_Y_path,params.batch_size)
    ds_val=get_data(params.val_X_path,params.val_Y_path,1)


    for i in range(params.epochs):
        loop = tqdm(ds_train, leave=True)
        print('test')
        test(ds_val, Dx, Dy, Gxy, Gyx)
        print('train')
        for x, y in loop:
            x, y = x[0][:, 0, ...], y[0][:, 0, ...]
            x, y = aug(x), aug(y)
            total_Gxy_loss,total_Gyx_loss,Dx_loss,Dy_loss = flow(x, y, l1, l2, Dx, Dy, Gxy, Gyx, Dx_optim, Dy_optim, Gxy_optim, Gyx_optim)
            loop.set_postfix(loss=f'total_Gxy_loss:{total_Gxy_loss},total_Gyx_loss:{total_Gyx_loss},Dx_loss:{Dx_loss},Dy_loss:{Dy_loss}')
        ckpt_manager.save()
        if i//1==0:
            print('test')
            test(ds_val,Dx, Dy, Gxy, Gyx)

if __name__=='__main__':
    main()


