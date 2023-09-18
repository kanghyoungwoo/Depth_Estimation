# PPB Depth-estimation

<aside>
👉 RGB 카메라로부터 입력된 이미지에 존재하는 탁구공을 검출하여 위치를 추정하는 프로젝트

</aside>

## **목차**

# 프로젝트 일정

---

![Untitled.png](PPB%20Depth-estimation%2054142c1d600c4d849bf81b9eba478c69/Untitled.png)

# Team Introduction

---

### 팀원 역할 소개

### 강형우

- 이미지 데이터 수집 및 라벨링
- Depth 추정 알고리즘 개발

### 조경수

- 이미지 데이터 수집 및 라벨링

### 한지호

- 자이카 패키지 구성
- 데이터 라벨링
- AWS관리 및 딥러닝 모델 학습

# Object Detection

---

### 객체 인식을 위한 적용 모델

- YOLOV3-tiny

### 학습데이터 구성

- 900장의 Train Data
- 약 100장의 Eval Data

### Labeling Tool

- Yololabel

### 데이터 증강(Data Augmentation) 적용 방법

- 어파인
- 밝기
- 색조
- 좌우반전
- 상하반전

### 모델 학습을 위한 하이퍼파라미터 설정 및 선정 이유

- 기본으로 설정된 파라미터 적용

### 학습 과정 및 결과

![KakaoTalk_20230208_203543827.png](PPB%20Depth-estimation%2054142c1d600c4d849bf81b9eba478c69/KakaoTalk_20230208_203543827.png)

800 epoch 진행

![KakaoTalk_20230220_132003462.png](PPB%20Depth-estimation%2054142c1d600c4d849bf81b9eba478c69/KakaoTalk_20230220_132003462.png)

![KakaoTalk_20230220_131949585.png](PPB%20Depth-estimation%2054142c1d600c4d849bf81b9eba478c69/KakaoTalk_20230220_131949585.png)

# Distance Estimation

---

### Camera Calibration 방법

- ROS Calibration Tool 이용

```python
rosrun camera_calibration cameraclibration.py --size 8x6 --square 0.0275 image:=/usb_cam/image_raw camera=/usb_cam
```

### Calibration Result

![cali.jpg](PPB%20Depth-estimation%2054142c1d600c4d849bf81b9eba478c69/cali.jpg)

### 객체의 위치(거리)추정 방법

### Geometrical Distance Estimation

- 종방향
    
    ![projection_1.jpg](PPB%20Depth-estimation%2054142c1d600c4d849bf81b9eba478c69/projection_1.jpg)
    
    ![projection_1_2.jpg](PPB%20Depth-estimation%2054142c1d600c4d849bf81b9eba478c69/projection_1_2.jpg)
    

![depth.jpg](PPB%20Depth-estimation%2054142c1d600c4d849bf81b9eba478c69/depth.jpg)

distance = H / y_norm

y_norm = H / distance

- 횡방향

d_x = tan(theta) * d_z

![projection_2.jpg](PPB%20Depth-estimation%2054142c1d600c4d849bf81b9eba478c69/projection_2.jpg)

![projection_2_2.jpg](PPB%20Depth-estimation%2054142c1d600c4d849bf81b9eba478c69/projection_2_2.jpg)

```python
def get_depth(bbox_list, camera_height, fy, cy):
    for index, bbox in enumerate(bbox_list):
        # normalized Image plane
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        y_norm = (ymax - cy) / fy

        delta_x = (xmax+xmin)/2 - 320
        azimuth = (delta_x/320) * FOV_H
        y_distance = 1 * camera_height / y_norm
        x_distance = 100 * (y_distance * math.tan(math.pi * (azimuth/180.)))
        y_distance = 100 * ((y_distance - 0.15) * 0.5877 + 2.0445275)
        distance = int(math.sqrt((x_distance * x_distance) + (y_distance * y_distance)))

        #distance = 1 * camera_height / y_norm
        # m -> cm
        bbox_list[index][4] = distance
        bbox_list[index][5] = x_distance
        bbox_list[index][6] = y_distance
    return bbox_list
```

### 거리 추정을 위한 추가적인 알고리즘 적용

![오차발생원인.jpg](PPB%20Depth-estimation%2054142c1d600c4d849bf81b9eba478c69/%25EC%2598%25A4%25EC%25B0%25A8%25EB%25B0%259C%25EC%2583%259D%25EC%259B%2590%25EC%259D%25B8.jpg)

Vision-based Acc with a Single Camera:: Bounds on Range and Range Rate Accuracy

- 이미지에서 픽셀은 정수인데, 실거리는 실수임에 따라서 거리 오차가 발행하였고, 타겟까지의 거리가 증가함에 따라 거리의 오차가 선형적으로 증가함. 따라서 단순 선형회귀분석을 통하여 오차 보정을 진행하였다.

### Code

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

device_name = tf.test.gpu_device_name()
if device_name != 'device:GPU:0':
    raise SystemError('GPU device not found')
else:
    print('Find Gpu at: {}'.format(device_name))

x_train = []
y_train = []
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.000001)

session = tf.Session()
session.run(tf.global_variables_initializer())

for i in range(100000001):
    session.run(train)
    if i % 100 == 0:
        print(i, session.run(cost), session.run(W), session.run(b))
```

![3.jpg](PPB%20Depth-estimation%2054142c1d600c4d849bf81b9eba478c69/3.jpg)

# 결론

### 탁구공 인식

- 총 6개의 탁구공 中 5개 인식 (흰 탁구공 3개, 주황 탁구공 2개)
    
    ![KakaoTalk_20230220_143142348.jpg](PPB%20Depth-estimation%2054142c1d600c4d849bf81b9eba478c69/KakaoTalk_20230220_143142348.jpg)
    
    ![KakaoTalk_20230220_143142348_01.jpg](PPB%20Depth-estimation%2054142c1d600c4d849bf81b9eba478c69/KakaoTalk_20230220_143142348_01.jpg)
    

### 거리 추정

- 흰 탁구공

x_dist = { 103.71224930974742}  
y_dist = { -6.22105696561771 } 

x_dist = { 169.28999320466596 } 
y_dist = { 62.270271171468295} 

x_dist = { 43.289170173710446 } 
y_dist = { -64.75536338669534 } 

- 주황색 탁구공

x_dist = { 42.54576854601101 } 
y_dist = { 53.32239852386017 } 
x_dist = {}
y_dist = {}
x_dist = { 179.3589350099815 } 
y_dist = { -86.03436199798273 } 

---
