#패키지 불러오기
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
%matplotlib inline

#train 셋 불러오기
labeled_images = pd.read_csv('c:/Users/stu/Desktop/number/train.csv')
images = labeled_images.iloc[0:40000,1:]
labels = labeled_images.iloc[0:40000,:1]

#random 으로 데이터 나누기
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)


#기존의데이터는 회색 ,즉 0~255로 명암의 정도를 나누어둠.
i=1
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))  #28x28짜리 픽셀
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])

plt.hist(train_images.iloc[i])

#훈련시키기
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())#############################
clf.score(test_images,test_labels)

#야예 1아니면 0으로 나눔 (0보다 큰값들은 전부 1로 표현)
test_images[test_images>0]=1
train_images[train_images>0]=1

#그림확인
img=train_images.iloc[i].as_matrix().reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])

plt.hist(train_images.iloc[i])

#다시 학습시키기
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)

#테스트셋을 불러와서 모델의 정확도를 확인        
test_data=pd.read_csv('c:/Users/stu/Desktop/number/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:5000])

results


df = pd.DataFrame(results)
df.index+=1
df.index.name='ImageId'
df.columns=['Label']
df.to_csv('results.csv', header=True)

df




