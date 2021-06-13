import cv2
import numpy as np
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

res_path='./train/generated_img/'
x_path='./data/base_data/testX/testA/'
y_path='./data/base_data/testY/testB/'

res_names=os.listdir(res_path)

x_names=os.listdir(x_path)
y_names=os.listdir(y_path)

data=[]
labels=[]

for res_name,name in zip(res_names,x_names+y_names):
    res_img=cv2.imread(f'{res_path}{res_name}').flatten()
    if res_name[0]=='X':
        img = cv2.imread(f'{x_path}{name}')
        label=0 #X
        mapping_label=2 #X2Y
    else:
        img=cv2.imread(f'{y_path}{name}')
        label=1 #Y
        mapping_label=3 #Y2X
    img=cv2.resize(img,(256,256)).flatten()
    data.append(img)
    labels.append(label)
    data.append(res_img)
    labels.append(mapping_label)
data=np.array(data)/255

labels=np.array(labels)

#pca=PCA(n_components=2)
#data=pca.fit_transform(data)

tsne=TSNE(n_components=2,random_state=0)
data=tsne.fit_transform(data)

X=data[np.where(labels==0)[0]]
X2Y=data[np.where(labels==2)[0]]
Y=data[np.where(labels==1)[0]]
Y2X=data[np.where(labels==3)[0]]

plt.scatter(X[:,0],X[:,1],c='r',label='X',edgecolor='black',marker='s',s=70)
plt.scatter(X2Y[:,0],X2Y[:,1],c='purple',label='X2Y',edgecolor='black',marker='s',s=70)
plt.scatter(Y[:,0],Y[:,1],c='yellow',label='Y',edgecolor='black',marker='s',s=70)
plt.scatter(Y2X[:,0],Y2X[:,1],c='orange',label='Y2X',edgecolor='black',marker='s',s=70)
plt.title('t-sne reduction')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend()
plt.savefig('tsne_result.jpg')
plt.show()



