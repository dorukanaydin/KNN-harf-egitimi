import cv2
import numpy as np
#Hazır data dosyasını kullanarak harf eğitip test eden ve doğruluk payını yazan program

#https://archive.ics.uci.edu/ml/index.php     data kaynakları(sayı,harf vb.)
#https://github.com/opencv/opencv/blob/4.x/samples/data/letter-recognition.data

#Data hazırlama
"""
Elimizdeki datada 0.indis harflerden oluşmakta bu yüzden float hatası almamamız için harflerin ascii tablosundaki değerlerini yazdık.
Ascii tablosunda a=65 dir ve harfler sırayla 1 artarak gider.Bu yüzden -65 yazdık.(b=66)
def x(a): return a+5    ==  x = lambda a: a+5   """

data = np.loadtxt(r"opencv\video_ve_resimler\letter-recognition.data", dtype="float32", delimiter=",", 
                  converters={0: lambda x: ord(x)-65})

train, test = np.vsplit(data,2)
#Burada 0.sütun(harfler) cevaplar, gerisi trainData olarak ayarlandı.
train_responses, trainData = np.hsplit(train, [1])  
test_responses, testData = np.hsplit(test, [1])

#Dataların eğitimi
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, train_responses)
ret, results, neighbours, distance = knn.findNearest(testData, 5) 

matches = test_responses == results
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / results.size

print("accuracy =",accuracy )