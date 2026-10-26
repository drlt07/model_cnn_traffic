import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

data = []
labels = []
classes = 43 #số lương thư mục ảnh
cur_path = os.getcwd()

# Truy xuất hình ảnh và nhãn của chúng
for i in range(classes):
    path = os.path.join(cur_path, 'train', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(os.path.join(path, a)).convert('RGB')
            image = image.resize((30, 30))
            image = np.array(image)

            data.append(image)
            labels.append(i)
        except:
            print("Load ảnh lỗi")

# Chuyển đổi danh sách thành mảng trống
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)
# Tách tập dữ liệu đào tạo và thử nghiệm
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Chuyển đổi các nhãn thành một mã hóa
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Xây dựng mô hình CNN cho bài toán này
'''model = Sequential()
width = 32
height = 32
classes = 43
shape = (width, height, 3)
model.add(Conv2D(32, (3, 3), padding="same", input_shape=shape))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dense(classes))
model.add(Activation("softmax"))
model.summary()'''

model = Sequential()
model.add(Conv2D(filters=32, padding='same', kernel_size=(5,5),
                 activation='relu',input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, padding='same', kernel_size=(5,5),
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, padding='same', kernel_size=(3, 3),
                 activation='relu'))
model.add(Conv2D(filters=64, padding='same', kernel_size=(3, 3),
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
model.summary()

aug = ImageDataGenerator(rotation_range=0.18, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
learning_rate = 0.01
epochs = 20
opt = SGD(learning_rate=learning_rate, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Start training")
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

model.save("traffic_classifier.h5") #lưu file train (gui.py đọc file này)

# vẽ đồ thị về độ chính xác của mô hình
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
# vẽ đồ thị về sự mất mát của mô hình
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Kiểm tra độ chính xác trên tập dữ liệu kiểm tra
from sklearn.metrics import accuracy_score, classification_report

y_test = pd.read_csv('Test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data = []

for img in imgs:
    image = Image.open(img).convert('RGB')
    image = image.resize((30, 30))
    data.append(np.array(image))

X_test = np.array(data)

pred = np.argmax(model.predict(X_test), axis=1)

# Độ chính xác của dữ liệu thử nghiệm
print('Test data accuracy:', accuracy_score(labels, pred))
# Precision, Recall, F1 của từng lớp và trung bình (macro avg)
print(classification_report(labels, pred, digits=2))

# Chuyển mô hình sang TensorFlow Lite để chạy trên điện thoại
# (dựng lại mô hình với kích thước lô = 1 để thư viện TFLite trên điện thoại đọc được)
m1 = Sequential()
m1.add(Input((30, 30, 3), batch_size=1))
for layer in model.layers:
    m1.add(layer.__class__.from_config(layer.get_config()))
m1.set_weights(model.get_weights())
open('model.tflite', 'wb').write(tf.lite.TFLiteConverter.from_keras_model(m1).convert())
