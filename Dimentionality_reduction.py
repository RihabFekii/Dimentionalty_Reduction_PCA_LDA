from PIL import Image
import glob 
import numpy as np 

from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 

path='C:\\Users\\Asus\\Desktop\\SUPCOM_2020\\ML\\TP4\\yalefaces\subject01.centerlight'

#partie1 
img= Image.open(path)
img1 = np.array(img) #Transformation d'une image en une matrice
img2 = img1.reshape(img1.shape[0]*img1.shape[1]) #Transformation d'une matrice en un vecteur ligne
paths= glob.glob('C:\\Users\\Asus\\Desktop\\SUPCOM_2020\\ML\\TP4\\yalefaces\\*')

data1=[]

for i in paths : 
    imgg= Image.open(i)
    imgg1 = np.array(imgg) 
    imgg2 = imgg1.reshape(imgg1.shape[0]*imgg1.shape[1]) 
    data1.append(imgg2)  #type liste
data=np.array(data1)  #type array


#PCA

pca = PCA(n_components=165)
x1= pca.fit_transform(data1)

label=np.repeat(range(1,16),11)  #label from the fichier data

"""model"""

x_train , x_test, y_train, y_test = train_test_split(x1,label,test_size=0.33,random_state=0)

"""classification """

clf=svm.SVC(kernel='linear', C=1 , decision_function_shape='ovo') 

clf.fit(x_train,y_train)

"""test""" 

y_pred_test=clf.predict(x_test)

""" accuracy """

acc=accuracy_score(y_test, y_pred_test)


#######################################################

pca = PCA(n_components=100)
x1= pca.fit_transform(data1)

label=np.repeat(range(1,16),11)  #label from the fichier data

"""model"""

x_train , x_test, y_train, y_test = train_test_split(x1,label,test_size=0.33,random_state=0)

"""classification """

clf=svm.SVC(kernel='linear', C=1 , decision_function_shape='ovo') 

clf.fit(x_train,y_train)

"""test""" 

y_pred_test=clf.predict(x_test)

""" accuracy """

acc2=accuracy_score(y_test, y_pred_test)

#######################################################

pca = PCA(n_components=25)
x1= pca.fit_transform(data1)

label=np.repeat(range(1,16),11)  #label from the fichier data

"""model"""

x_train , x_test, y_train, y_test = train_test_split(x1,label,test_size=0.33,random_state=0)

"""classification """

clf=svm.SVC(kernel='linear', C=1 ) 

clf.fit(x_train,y_train)

"""test""" 

y_pred_test=clf.predict(x_test)

""" accuracy """

acc3=accuracy_score(y_test, y_pred_test)

#######################################################

pca = PCA(n_components=15)
x1= pca.fit_transform(data1)

label=np.repeat(range(1,16),11)  #label from the fichier data

"""model"""

x_train , x_test, y_train, y_test = train_test_split(x1,label,test_size=0.33,random_state=0)

"""classification """

clf=svm.SVC(kernel='linear', C=1 ) 

clf.fit(x_train,y_train)

"""test""" 

y_pred_test=clf.predict(x_test)

""" accuracy """

acc4=accuracy_score(y_test, y_pred_test)


""" LDA """

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
model = LDA()
x2= model.fit_transform(data,label)



"""model"""

x_train , x_test, y_train, y_test = train_test_split(x2,label,test_size=0.33)

"""classification """

 
model.fit(x_train,y_train)

"""test""" 

y_pred_test=model.predict(x_test)

""" accuracy """

acc_LDA=accuracy_score(y_test, y_pred_test)
    


