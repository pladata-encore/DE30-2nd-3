# 폐암 예측 모델

### 사용언어
![image](https://github.com/pladata-encore/DE30-2nd-3/assets/127280706/cc22352a-34fc-4c21-b84f-7ff13d558a6b)

### 사용 라이브러리
   - `Image`, `torch`, `nn`, `optim`, `lr_scheduler`, `transforms`
   - `pandas`, `numpy`
   - `matplotlib`, `plt`, `time`, `random`
   - `torchvision `, `datasets`, `train_test_split`, `random_split`
   -  `os`,  `glob`, `collections`
   -  `Sequential`, `Conv2D`, `AvgPool2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`

### 작업 툴
- Jupyter Lab
- Google Colab

# 프로젝트 개요
## 👑프로젝트 소개
흉부 CT 사진을 기반으로 폐암 여부를 진단하고, 폐암일 경우 암종(선암종, 대세포암종, 편평세포암종)을 구분해 명시해주는 서비스입니다.     
해당 서비스를 구현하기 위해 총 3개의 모델을 시도해봤습니다.   
1. Pytorch를 사용한 폐암 예측 모델
2. Tensorflow/Keras를 사용한 폐암 예측 모델
3. 3d 자료를 기반으로 한 폐 종양 감지 모델
## 💫주제 선정 배경 및 기대효과
현재 전공의 사직과 의료진 파업 때문에 많은 혼란이 발생하고 있습니다.   
의료 인력 부족으로 수술과 진료가 연기되거나 취소되는 경우가 속출하고 있어 이는 환자들의 치료 기회를 박탈하고 건강을 위협하는 결과를 초래하고 있습니다. 그래서 이런 상황에 조금이라도 도움이 되고자 이런 주제를 선택하게 되었습니다.    
폐암은 특히 한국에서 발생률과 사망률이 높은 질병 중 하나입니다. 조기 발견이 어려워 진단 시점에서 이미 진행된 상태인 경우가 많습니다.  이 예측 모델을 활용하면 조기 발견을 통해 환자의 생존율을 높일 수 있습니다. 또한 조기 진단과 치료는 질병이 진행된 후 치료하는 것보다 비용 효율적입니다. 의료진의 진단 과정을 지원하여 더 많은 환자를 더 정확하게 진료할 수 있게 합니다. 이는 의료진의 업무 만족도와 효율성을 높일 것입니다.

# 데이터 수집

# 모델 1
## Pytorch를 사용한 폐암 예측 모델

# 모델 2 
## Tensorflow/Keras를 사용한 폐암 예측 모델

- 구글 드라이브 연결

```python
from google.colab import drive
drive.mount('/content/drive')
```

- 필요한 라이브러리 import

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import PIL as pl
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AvgPool2D, MaxPooling2D, Flatten, Dense, Dropout
import zipfile
import os
```

- 데이터 준비

```python
import os

# 일반 폴더에서 불러오기

# Directory containing image files
img_dir = '/content/drive/MyDrive/부트캠프/프로젝트2/자료'

img_width = 128
img_height = 128

# Second section of the path
categories = ['편평세포암종', '선암종', '대세포암종', '정상']

# Create directories for extracted images
output_dir = '/content/img_data'
os.makedirs(output_dir, exist_ok=True)

# Extract images from the directories and store them in img_data folder
img_data = []
for idx, cata in enumerate(categories):
    folder = os.path.join(img_dir, cata)

    # Label for the current category
    label = idx

    # Check if the directory exists
    if os.path.exists(folder):
        # Check if images exist in the folder
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            try:
                # Attempt to read and resize the image
                img_array = cv2.imread(img_path)
                img_array = cv2.resize(img_array, (img_height, img_width))

                # Check if the image array is not empty
                if img_array is not None and not img_array.size == 0:
                    img_data.append([img_array, label])

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    else:
        print(f"Folder {folder} does not exist.")

# Now img_data contains all the images from the specified directories along with their labels.

random.shuffle(img_data)
```

```python
x=[]
y=[]
for features,labels in img_data:
    x.append(features)
    y.append(labels)

#Convert X and Y list into array
X=np.array(x, dtype = float)
Y=np.array(y, dtype = float)
```

- 정규화

```python
for i in range(len(X)):
    X[i] = X[i]/255.0
```

- X 형태 확인

```python
X.shape
```

- 데이터 분할

```python
x, x_test, y, y_test = train_test_split(X, Y, test_size = 0.2)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2)
```

- 데이터 증강

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,      # Rotate the image by up to 15 degrees
    width_shift_range=0.1,  # Shift the image horizontally by up to 10% of the width
    height_shift_range=0.1, # Shift the image vertically by up to 10% of the height
    shear_range=0.1,        # Shear the image by up to 10 degrees
    zoom_range=0.1,         # Zoom in or out by up to 10%
    horizontal_flip=True,   # Flip the image horizontally
    fill_mode='nearest'     # Fill in missing pixels with the nearest value
)

# Define the number of augmented images to generate per original image
augmented_images_per_original = 6

# Define the batch size
batch_size = 4

# 전체 데이터를 두 개의 그룹으로 나눔
total_data = len(x_train)
split_index = total_data // 2
x_train_group1, x_train_group2 = x_train[:split_index], x_train[split_index:]
y_train_group1, y_train_group2 = y_train[:split_index], y_train[split_index:]

# 각 그룹에 대해 데이터 증강 적용 및 결합
x_train_augmented, y_train_augmented = [], []

for x_train_group, y_train_group in [(x_train_group1, y_train_group1), (x_train_group2, y_train_group2)]:
    for i in range(0, len(x_train_group), batch_size):
        batch_x = x_train_group[i:i+batch_size]
        batch_y = y_train_group[i:i+batch_size]

        # 각 배치에 대해 데이터 증강 적용
        augmented_x_batch, augmented_y_batch = [], []
        for j in range(len(batch_x)):
            for _ in range(augmented_images_per_original):
                augmented_image = datagen.flow(np.expand_dims(batch_x[j], axis=0), batch_size=1)[0][0]
                augmented_x_batch.append(augmented_image)
                augmented_y_batch.append(batch_y[j])

        x_train_augmented.extend(augmented_x_batch)
        y_train_augmented.extend(augmented_y_batch)

# 데이터 결합
x_train_augmented = np.array(x_train_augmented)
y_train_augmented = np.array(y_train_augmented)

```

- 모델 정의 (CNN)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(128, (3, 3), padding='same', input_shape=X.shape[1:], activation='relu'))
model.add(AveragePooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dropout(0.2, seed=12))
model.add(Dense(3000, activation='relu'))
model.add(Dense(1500, activation='relu'))
model.add(Dense(4, activation='softmax'))  # 출력 층의 뉴런 수를 4로 변경

model.summary()
```

## CNN(컨볼루션 신경망)

CNN(Convolutional Neural Network)는 합성곱신경망으로도 불립니다. 주로 시각적 이미지를 분석하는 데 사용되는데 머신러닝의 한 유형인 딥러닝에서 가장 많이 사용되고 있는 알고리즘입니다.

CNN은 이미지 전체를 작은 단위로 쪼개어 각 부분을 분석하는데, 이는 이미지를 인식하기 위한 패턴을 찾는 데 유용합니다. 데이터를 통해 특징을 스스로 학습하고, 패턴을 사용하여 이미지를 분류합니다. 또한 특징을 수동으로 추출할 필요가 없습니다. 더불어 기존 네트워크를 바탕으로 새로운 인식 작업을 위해 CNN을 재학습하여 사용하는 것이 가능합니다.

![image](https://github.com/pladata-encore/DE30-2nd-3/assets/127280706/6273e337-d062-476f-801f-2ba40368a8ee)


- 모델 코드 상세 설명
1. **입력 레이어 (Input Layer)**:
    - 입력 이미지의 형태를 지정합니다. **`X.shape[1:]`**의 형태를 따르며, 이미지의 크기와 채널 수를 지정합니다.
2. **컨볼루션 레이어 (Convolutional Layers)**:
    - **`Conv2D`** 레이어를 사용하여 각 레이어는 컨볼루션 필터를 통해 이미지의 특징을 추출합니다.
    - 각 레이어에서는 커널 크기가 3x3이고, 활성화 함수는 ReLU(Rectified Linear Unit)를 사용합니다.
    - 첫 번째 컨볼루션 레이어는 128개의 필터를 가지며, 입력 이미지와 동일한 크기의 출력을 생성합니다. AveragePooling2D 레이어를 통해 공간 크기를 절반으로 줄입니다.
3. **풀링 레이어 (Pooling Layers)**:
    - **`MaxPooling2D`**와 **`AveragePooling2D`** 레이어를 사용하여 공간적인 차원을 줄입니다.
4. **플래튼 레이어 (Flatten Layer)**:
    - 다차원의 출력을 1차원으로 변환하여 완전 연결 레이어에 전달합니다.
5. **드롭아웃 레이어 (Dropout Layer)**:
    - **`Dropout`** 레이어를 사용하여 과적합을 방지하기 위해 학습 과정 중에 무작위로 일부 뉴런을 비활성화합니다.
6. **완전 연결 레이어 (Dense Layers)**:
    - **`Dense`** 레이어를 사용하여 이미지의 특징을 분류합니다.
    - 각 완전 연결 레이어는 3000개와 1500개의 뉴런을 가집니다.
    - 마지막 레이어는 출력 레이어로서, 4개의 클래스에 대한 확률을 출력하기 위해 Softmax 활성화 함수를 사용합니다.

- 조기종료 설정

```python

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

```

- 훈련

```python

history = model.fit(x_train, y_train, 
										validation_data = (x_val, y_val),epochs = 45)
```

![image](https://github.com/pladata-encore/DE30-2nd-3/assets/127280706/ca7aa4b3-efa0-479a-822c-91d7e6fb76da)


- 성능 평가

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
```

![image](https://github.com/pladata-encore/DE30-2nd-3/assets/127280706/1b27d5aa-533a-47b8-9d64-6f33b24b0aee)


- accuracy 그래프

```python
plt.plot(history.history['accuracy'], label = 'Train accuracy')
plt.plot(history.history['val_accuracy'], label = 'val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc = 'best')
```

![image](https://github.com/pladata-encore/DE30-2nd-3/assets/127280706/9d08d40d-d7d9-427a-9952-6c436f374276)


이 그래프는 모델의 학습 및 검증 과정에서의 정확도(accuracy) 변화를 보여줍니다. 파란색 선은 훈련 데이터(train accuracy)에 대한 정확도 변화를 나타내고, 주황색 선은 검증 데이터(validation accuracy)에 대한 정확도 변화를 나타냅니다. 

### **1. 초기 단계 (10 epoch 이하)**

- **빠른 정확도 증가**: 처음 몇 epoch 동안 훈련 데이터와 검증 데이터에 대한 정확도가 빠르게 증가합니다. 이는 모델이 초기 학습 단계에서 중요한 패턴을 빠르게 학습하고 있음을 의미합니다.

### **2. 중간 단계 (대략 10-30 epoch)**

- **변동성**: 이 단계에서는 정확도가 약간의 변동을 보입니다. 훈련 데이터에 대한 정확도는 비교적 일정하게 증가하지만, 검증 데이터에 대한 정확도는 약간의 진폭을 가지고 변동합니다. 이는 모델이 데이터를 더 깊이 학습하면서 일반화 성능을 조정하고 있음을 나타냅니다.

### **3. 후반 단계 (30 epoch 이후)**

- **안정화**: 훈련 데이터에 대한 정확도는 거의 100%에 도달하며 안정화됩니다. 검증 데이터에 대한 정확도도 높은 수준에서 안정화되지만 훈련 데이터에 비해 약간 낮습니다.

### **과적합(Overfitting)**

- 과적합은 모델이 훈련 데이터의 패턴을 지나치게 학습하여 검증 데이터나 새로운 데이터에 대해 잘 일반화하지 못하는 현상입니다.
- 저희가 사용한 모델은 훈련 데이터에 대한 정확도가 거의 100%에 도달했지만, 검증 데이터에 대한 정확도는 그보다 낮습니다. 이는 모델이 훈련 데이터에 과적합(overfit)되었음에 대한 가능성을 열어둘 수 있습니다.

### **모델 성능**

- **전반적인 성능**: 모델은 훈련과 검증 데이터 모두에서 높은 정확도를 달성하고 있으며, 이는 모델이 주어진 데이터에 대해 좋은 성능을 보여줍니다.
- **검증 정확도**: 검증 데이터에 대한 정확도가 지속적으로 높다는 것은 모델이 새로운 데이터에 대해서도 비교적 잘 작동할 가능성이 높음을 시사합니다.

### **결론**

이 그래프는 모델이 학습 데이터에 대해 매우 잘 학습하고 있지만, 검증 데이터의 정확도는 높지만 훈련 데이터보다 낮기 때문에 약간의 과적합이 발생하고 있음을 보여줍니다. 

- Loss 그래프

```python
plt.plot(history.history['loss'], label = 'Train Loss')
plt.plot(history.history['val_loss'], label = 'Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc = 'best')
```

![image](https://github.com/pladata-encore/DE30-2nd-3/assets/127280706/becb6715-9b9c-4d45-b37d-e23efe6701d3)


이 그래프는 모델의 학습 및 검증 과정에서의 손실(loss) 변화를 보여줍니다. 파란색 선은 훈련 데이터(train loss)에 대한 손실 변화를 나타내고, 주황색 선은 검증 데이터(validation loss)에 대한 손실 변화를 나타냅니다. 

### **1. 초기 단계 (10 epoch 이하)**

- **빠른 손실 감소**: 처음 몇 epoch 동안 훈련 데이터와 검증 데이터에 대한 손실이 빠르게 감소합니다. 이는 모델이 초기 학습 단계에서 중요한 패턴을 빠르게 학습하고 있음을 의미합니다.

### **2. 중간 단계 (대략 10-30 epoch)**

- **변동성 증가**: 이 단계에서 검증 손실(val loss)은 상당한 변동을 보입니다. 특히 epoch 20 이후 검증 손실이 급격히 증가하고 감소하는 패턴을 반복합니다. 이는 모델이 과적합(overfitting)되고 있음을 시사합니다.
- **훈련 손실 감소**: 반면, 훈련 손실(train loss)은 계속해서 감소하고 있으며, 거의 0에 근접합니다. 이는 모델이 훈련 데이터에 대해 매우 잘 학습하고 있음을 나타냅니다.

### **3. 후반 단계 (30 epoch 이후)**

- **훈련 손실 안정화**: 훈련 손실은 거의 0에 도달하며 안정화됩니다.
- **검증 손실 변동 지속**: 검증 손실은 여전히 변동이 크며, 특정 지점에서는 급격히 증가합니다.

### **과적합(Overfitting)**

- 훈련 손실이 거의 0에 도달한 반면, 검증 손실이 높고 불안정하게 변동하고 있습니다.  과적합의 징후로 볼 수 있습니다.

### **모델 성능**

- **훈련 데이터에 대한 성능**: 모델은 훈련 데이터에 대해 매우 잘 학습하고 있으며, 훈련 손실이 거의 0에 도달했습니다.
- **검증 데이터에 대한 성능**: 검증 데이터에 대한 손실은 훈련 데이터에 비해 높고 변동성이 큽니다. 이는 모델이 새로운 데이터에 대해 안정적으로 성능을 내지 못할 수 있음을 시사합니다.

### **결론**

이 그래프는 모델이 훈련 데이터에 대해 매우 잘 학습하고 있지만, 검증 데이터에 대해서는 약간의 과적합(overfitting)현상에 대한 징후가 나타나고 있음을 보여줍니다.

- 이미지 입력해보기

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the test image
image_path = '/content/drive/MyDrive/부트캠프/프로젝트2/test_image.jpg'
img = cv2.imread(image_path)

img = cv2.resize(img, (img_height, img_width))

# Expand dimensions to create a batch of size 1
img = np.expand_dims(img, axis=0)

# Make prediction
prediction = model.predict(img)

# Interpret prediction
class_label = np.argmax(prediction)
confidence = prediction[0][class_label]

print("Predicted Class Label:", class_label)
print("Confidence:", confidence)

# Mapping class labels to categories
class_labels_to_categories = {0: 'squamous cell carcinoma', 1: 'adenocarcinoma', 2: 'Large cell carcinoma', 3: 'normal'}
# ['편평세포암종', '선암종', '대세포암종', '정상']
# Display the predicted category
predicted_category = class_labels_to_categories[class_label]
print("Predicted Category:", predicted_category)

img = cv2.imread(image_path)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(predicted_category)
plt.show()
```

![image](https://github.com/pladata-encore/DE30-2nd-3/assets/127280706/8ac367c4-ab4c-4a35-b1e7-19e3feb19e11)


## 제 1종 오류, 2종 오류

1종 오류(Type I error)와 2종 오류(Type II error)는 통계학에서 사용되는 용어로, 주로 가설 검정에서 사용됩니다. 혼동 행렬(confusion matrix)에서는 다음과 같이 해석할 수 있습니다:

- **1종 오류 (Type I error)**: 실제로는 음성(negative)인데 양성(positive)으로 잘못 예측하는 경우. 즉, 거짓 양성(False Positive, FP).
- **2종 오류 (Type II error)**: 실제로는 양성(positive)인데 음성(negative)으로 잘못 예측하는 경우. 즉, 거짓 음성(False Negative, FN).

암과 같은 질병을 예측하는 모델에서는 질병이 '양성', 질병이 없는 상태가 '음성'으로 정의됩니다.

```python
result = model.predict(x_test)
```

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define true labels and predicted labels
true_labels = y_test
predicted_labels = np.argmax(result, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels_to_categories.values(),
            yticklabels=class_labels_to_categories.values())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
```

![image](https://github.com/pladata-encore/DE30-2nd-3/assets/127280706/0bfe2fbf-6674-420f-a4dc-f0fbc7bbe5c9)


이 혼동 행렬(confusion matrix)은 모델이 네 가지 클래스를 예측한 결과를 보여줍니다: 편평 세포 암종(squamous cell carcinoma), 선암(adenocarcinoma), 대세포 암종(large cell carcinoma), 정상(normal)입니다. 각 행은 실제 라벨을, 각 열은 모델의 예측 라벨을 나타냅니다.

### **편평세포암종  (Squamous cell carcinoma)**

- **1종 오류 (False Positive, FP)**: 실제로는 편평 세포 암종이 아닌데 편평 세포 암종으로 예측한 경우.
    - 다른 클래스의 샘플이 편평 세포 암종으로 잘못 예측된 경우: 5 + 2 + 0 = 7
- **2종 오류 (False Negative, FN)**: 실제로는 편평 세포 암종인데 다른 클래스로 예측한 경우.
    - 편평 세포 암종 샘플이 다른 클래스로 잘못 예측된 경우: 9 + 2 + 1 = 12

### **선암 (Adenocarcinoma)**

- 실제로 선암인 44개의 샘플 중 31개는 정확하게 선암으로 예측되었고, 5개는 편평 세포 암종으로, 7개는 대세포 암종으로, 1개는 정상으로 잘못 예측되었습니다.
- **1종 오류 (False Positive, FP)**: 실제로는 선암이 아닌데 선암으로 예측한 경우.
    - 다른 클래스의 샘플이 선암으로 잘못 예측된 경우: 9 + 1 + 0 = 10
- **2종 오류 (False Negative, FN)**: 실제로는 선암인데 다른 클래스로 예측한 경우.
    - 선암 샘플이 다른 클래스로 잘못 예측된 경우: 5 + 7 + 1 = 13

### **대세포 암종 (Large cell carcinoma)**

- 실제로 대세포 암종인 29개의 샘플 중 25개는 정확하게 대세포 암종으로 예측되었고, 2개는 편평 세포 암종으로, 1개는 선암으로, 1개는 정상으로 잘못 예측되었습니다.
- **1종 오류 (False Positive, FP)**: 실제로는 대세포 암종이 아닌데 대세포 암종으로 예측한 경우.
    - 다른 클래스의 샘플이 대세포 암종으로 잘못 예측된 경우: 2 + 7 + 1 = 10
- **2종 오류 (False Negative, FN)**: 실제로는 대세포 암종인데 다른 클래스로 예측한 경우.
    - 대세포 암종 샘플이 다른 클래스로 잘못 예측된 경우: 2 + 1 + 1 = 4

### **정상 (Normal)**

- 실제로 정상인 99개의 샘플 중 98개는 정확하게 정상으로 예측되었고, 1개는 대세포 암종으로 잘못 예측되었습니다.
- **1종 오류 (False Positive, FP)**: 실제로는 정상이 아닌데 정상으로 예측한 경우.
    - 다른 클래스의 샘플이 정상으로 잘못 예측된 경우: 1 + 1 + 1 = 3
- **2종 오류 (False Negative, FN)**: 실제로는 정상인데 다른 클래스로 예측한 경우.
    - 정상 샘플이 다른 클래스로 잘못 예측된 경우: 1

모델의 성능을 요약하자면:

- 정상 샘플에 대해 매우 높은 정확도를 보이고 있으며, 정상 샘플 중 하나만 잘못 예측되었습니다.
- 암종에 대해서는 비교적 높은 정확도를 보이지만, 클래스 간의 혼동이 발생하고 있습니다. 특히 편평 세포 암종과 선암 사이의 혼동이 비교적 자주 발생합니다.

**따라서 해당 모델은 정상과 비정상을 구분하는 데 매우 강하지만, 암의 세부 유형을 구분하는 데는 약간의 개선이 필요합니다.**

**특히 의료 분야에서는 2종 오류(질병을 놓치는 경우)가 더 치명적일 수 있으므로 이를 최소화하는 것이 향후 저희 팀이 해당 모델을 발전시키는 데 중요한 목표가 될 것입니다.**


