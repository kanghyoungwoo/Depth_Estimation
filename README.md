# Object Detection

<aside>
👉 딥러닝을 통해 표지판과 신호등을 학습한 후 딥러닝 모델을 포팅한 모형차량을 차선인식을 통해 자율주행을 합니다.

</aside>

## **목차**

# Team Introduction

---

### 팀원 역할 소개

### 강형우

- 이미지 데이터 수집 및 라벨링
- 신호등 신호 추출 알고리즘
- 정지선 인식 알고리즘

### 한지호

- 이미지 데이터 수집 및 라벨링
- 모델학습
- 차선인식 및 주행제어

# Object Detection

---

### 객체 인식을 위한 적용 모델

- YOLOV3-tiny
    
    자이카에 포팅 가능한 YOLOv3와 YOLOV3-tiny 둘 중 real-time에 적용하려면 보다 가벼운 모델이 더 적합하다고 판단되어 YOLOV3-tiny 모델을 선정하게 되었습니다.
    

### 학습 데이터 구성

### train, validation 데이터

- (클래스별) 학습 데이터 수: 4500장 (균등하게 모든 클래스를 맞춤)
- (클래스별) 평가 데이터 수: 500 장(평가를 위한 데이터를 따로 만듦)
- 검수

### 데이터 수집 방법

- 자이카 카메라로 동영상을 찍은 후 10프레임 단위로 끊어서 이미지 생성

### Labeling Tool

- Yololabel

### Labeling 기준

- Yololabel
- 우회전, 좌회전은 멀어져서 화살표 부분에서 직각 부분이 안보이면 ignore 처리
- 횡단보도는 글자가 너무 작아져서 보이지 않는 경우 ignore 처리
- 표지판끼리 겹칠 경우 50%이상이면 박스를 안침
- 화살표의 경우 기둥이나 방향이 모두 잘리면 박스를 안침

### 사용한 데이터 증강(Data Augmentation)

- 어파인
- 밝기
- 색조
- 좌우반전
- 모션블러

### 모델 학습을 위한 하이퍼파라미터 설정 및 선정 이유

- 기본으로 설정된 파라미터 적용
- Optimizer를 adam으로 바꾸어 보았는데 정확도가 더 떨어져 기본으로 제공된 SGD를 이용

### 학습 과정 및 결과

- 1일차 - 데이터 2500장 정도 수집. 데이터 라벨링 기준을 정하고 라벨링 시작
- 2일차 - 데이터 라벨링을 하면서 유난히 적은 데이터를 보완 + 어두운 환경에서 추가 데이터 수집. 어느정도 라벨링 된 데이터로 학습 시작
- 3일차 - 학습 결과를 자이카에 직접 포팅해서 결과를 확인 → 흔들릴때마다 인식률이 떨어지는것을 확인 (아래 영상 참고)
- 
- 4일차 - 데이터 라벨링 마무리 + 검수 진행 (해당 과정에서 잘못된 라벨링을 많이 수정함) 정제된 데이터로 다시 학습 시작
- 5일차 - 기존데이터로 학습을 마무리 + 흔들릴때마다 인식률이 떨어지는 것을 고려하여 모션블러 어그멘테이션 추가
- 2일동안 1650 epoch 학습을 진행
- 자이카에 포팅한 최종 결과

### Loss, accuracy 그래프

### 사진

![KakaoTalk_20230905_154402533.jpg](Object%20Detection%20212b67471a484b2d9fd0e0be829cd663/KakaoTalk_20230905_154402533.jpg)

![KakaoTalk_20230905_154402533_01.jpg](Object%20Detection%20212b67471a484b2d9fd0e0be829cd663/KakaoTalk_20230905_154402533_01.jpg)

# 어려웠던 부분 & 개선 방법

---

### 1. 교차로 구간에서 차선 인지

- 일단 먼저 직선 구간에서는 조향각이 크게 변할 필요가 없기 때문에 직선과 곡선구간의 PID값을 다르게 설정했다. 그리고 기존 차선 인식 코드는 차선을 잃으면 초기화되는 코드가 들어있었는데 해당 코드를 그대로 사용하면 교차로에서 왼쪽이나 오른쪽에서 먼저 초기화가 되면 방향을 확 틀어버린 채로 직진을 해버리는 문제가 있었다. 그래서 차선을 잃으면 기존차선을 유지하도록 코드를 변경하였다.
- 이와 추가로 수평에 가까운 선이 인지되는 경우 급격한 기울기로 인해 lpos와 rpos가 교차되는 경우도 빈번히 일어났는데 이 부분에 대한 필터링도 진행하였다. 교차되는 경우를 그대로 두면 average filter에 들어가서 평균값이 확 뒤틀리는 일이 발생한다.

```cpp
std::pair<int, int> HoughTransformLaneDetector::refine_LanePosition(int lpos, int rpos)
{
  if ((lpos <= 0) && (rpos < image_width_))
  {
    lpos = rpos - 300;
    l_samples_.clear();
    addRSample(rpos);
    right_mean = getRWeightdMovingAverage();
    lpos_flag = false;
    rpos_flag = true;
  }
  else if ((lpos > 0) && (rpos == image_width_))
  {
    rpos = lpos + 380;
    r_samples_.clear();
    addLSample(lpos);
    left_mean = getLWeighteedMovingAverage();
    lpos_flag = true;
    rpos_flag = false;
  }
  else if (lpos > rpos)
  {
    lpos = left_mean;
    rpos = right_mean;
    addLSample(lpos);
    addRSample(rpos);
    left_mean = getLWeightedMovingAverage();
    right_mean = getRWeightedMovingAverage();
  }
  else if ((lpos <= 0) && (rpos >= image_width_))
  {
    l_samples_.clear();
    r_samples_.clear();
    lpos = left_mean;
    rpos = right_mean;
    lpos_flag = false;
    rpos_flag = false;
  }
  else ()
  {
    addLSample(lpos);
    addRSample(rpos);
    left_mean = getLWeightedMovingAverage();
    right_mean = getRWeightedMovingAverage();
    lpos_flag = true;
    rpos_flag = true;
  }
  return std::pair<int, int>(lpos, rpos);
}
```

### 2. 합류 구간에서 차선 인지

- 합류구간에서 제대로 된 차선을 인지하기 힘들었다. 기울기가 0이 되어 rpos와 lpos가 교차되는 일도 발생하고 원하는 값과 거리가 먼 값들이 나오는 경우도 더러 있었다. 이를 해결하기 위해 차선만 이용하는 방법을 택했다. 제어에 자신이 있었기 때문에 제어값만 찾는다면 훨씬 더 안정적인 주행이 가능할 것이라 보았다.

```cpp
void LaneKeepingSystem::run()
{
  int lpos, rpos, error, ma_mpos;
  float steering_angle;
  while (ros::ok())
  {
    ros::spinOnce();
    if (frame_.empty() || (bbox_flag_ == false))
    {
      continue;
    }
    if (object_id_ == 0)
    {
      drive_normal("left", 15.0);
      stop_count_ = 0;
    }
    else if (object_id_ == 1)
    {
      drive_normal("right", 15.0);
      stop_count_ = 0;
    }
    else if ((object_id_ == 2) || (object_id_ == 3))
    {
      drive_stop("straight", 6.0, 2.0);
    }
    else if (object_id_ == 4)
    {
      traffic_light_ = traffic_sign_detect(frame_, xmin_, xmax_, ymin_, ymax_)
      if (traffic_light_ == false)
      {
        ++red_count_;
        if (red_count_ > 3)
        {
          drive_stop("straight", 2.0, 0.0);
          green_count_ = 0;
        }
      }
      else
      {
        ++green_count_;
        if (green_count_ > 6)
        {
          drive_normal("go", 0.0);
          red_count_ = 0;
        }
      }
    }
    else
    {
      drive_normal("go", 0.0);
    }
    object_id_ = -1;
  }
}
```

### 3. PID 제어

- 직선 구간에서는 조향각이 크게 변할 필요가 없기 때문에 직선과 곡선 구간의 PID값을 다르게 설정했다. 그리고 빠른 반응속도를 위해 error가 0인 순간에 i_error를 0으로 초기화 해주었다.

```cpp
float PID::getControlOutput(int error)
{
  if (current_angle < 20.0F)
  {
    p_gain_ = 0.30F;
    i_gain_ = 0.00000F;
    d_gain_ = 0.00F;
  }
  else
  {
    p_gain_ = 0.37F;
    i_gain_ = 0.000025F;
    d_gain_ = 0.00F;
  }
  
  float float_type_error = static_cast<float>(error);
  if (error == 0)
  {
    i_error_ = 0;
  }
  else
  {
    i_error_ += float_type_error;
  }
  p_error_ = float_type_error;
  d_error_ = float_type_error - p_error_;
  return p_gain_ * p_error + i_gain_ * i_error_ + d_gain_ * d_error_;
}
```

# 신호등 색 분류 방법

---

### 분류 방법

- 신호등 분류 방법은 HSV 색 공간 검출이라는 방법을 사용했고, 그 안에서 2가지로 나누어 2 방법을 모두 사용해 보았습니다.

### 1. HoughCircle함수 이용

첫 번째 방법으로는 OpenCV의 HoughCirle함수를 이용했습니다. 신호등BoungdingBox를 crop하여 HSV를 이용한 색상이 검출되면 binary 이미지를 mask로 사용하여 원본 이미지에서 범위 값에 해당되는 부분에 흰색이 신호등 원 모양으로 동그랗게 찍히게 되면 HoughCircle에서 원이 검출이 되고, 검출된 원이 특정 색상의 원이므로 원이 검출될때마다 특정 색상의 id값을 return 해주었습니다. 하지만 문제점은, 자이카가 신호등 가까이에 갔을 때 조명이 신호등 불빛에 반사가 되어 검출되어야 하는 색상이 아닌 다른 곳에서 mask된 원본 이미지에서 해당 부분에 흰색이 찍히게 되었고, 그 결과 다른 색 원이 검출되어 검출해야할 색상의 id가 아닌 다른 색상의 id를 return하는 문제가 발생했습니다. 간헐적으로만 발생한다면 문제가 없을 것이지만 다른 색상 검출이 꽤 잦은 빈도로 나타나 2번째 방법을 사용했습니다. 

### 2. 픽셀 수 계산 방법

- 두 번째 방법은 BoundingBox의 신호등BoungdingBox를 crop하여 HSV를 이용한 색상이 검출되면 binary 이미지를 mask로 사용하여 원본 이미지에서 범위 값에 해당되는 부분에서의 픽셀 값 개수를 계산했습니다. 이렇게 되면 이 전에 발생했던 가끔 다른 색이 튀어도, 본래 검출하고자 하는 색상의 색이 훨씬 검출이 잘 되므로 픽셀 값 개수가 많을 것이고 따라서 첫 번째 방법보다 훨씬 더 정확도가 높은 검출율을 보여주었습니다.

```cpp
bool LaneKeepingSystem::traffic_sign_detect(cv::Mat frame, int xmin, int xmax, int ymin, int ymax)
{
  xmin = std::min(416, std::max(xmin, 0));
  xmax = std::min(416, std::max(xmax, 0));
  ymin = std::min(416, std::max(ymin, 0));
  ymax = std::min(416, std::max(ymax, 0));
  double height_resize = 480.0 / 416.0;
  double width_resize = 640.0 / 416.0;
  xmin = static_case<int>(xmin * width_resize);
  xmax = static_case<int>(xmax * width_resize);
  ymin = static_case<int>(ymin * width_resize);
  ymax = static_case<int>(ymax * width_resize);
  cv::Rect rect(xmin, ymin, xmax - xmin, ymax - ymin);
  cv::Mat cropped_img = frame(rect);
  cv::Mat hsv_img;
  cv::cvtColor(cropped_img, hsv_img, cv::COLOR_BGR2HSV);

  cv:: Mat red_mask, red_image;
  cv:: Mat green_mask, green_image;

  cv:: Scalar lower_red = cv::Scalar(160, 80, 80);
  cv:: Scalar upper_red = cv::Scalar(180, 255, 255);

  cv:: Scalar lower_green = cv::Scalar(60, 80, 100);
  cv:: Scalar upper_green = cv::Scalar(100, 255, 255);

  cv::inRange(hsv_img, lower_red, upper_red, red_mask);
  cv::inRange(hsv_img, lower_green, upper_green, green_mask);

  cv::Mat dst;
  cv::bitwise_and(hsv_img, hsv_img, red_image, red_mask);
  cv::bitwise_and(hsv_img, hsv_img, green_image, green_mask);

  int red_result = 0;
  int green_result = 0;
  int result = 0;

  for (int y = 0;, y < ymax - ymin; ++y)
  {
    for(int x = 0; x < xmax - xmin; ++x)
    {
      red_result += red_mask.at<uchar>(y, x);
      green_result += green_mask.at<uchar>(y, x);
    }
  }
  if (green_result > red_result)
  {
    return true;
  }
  else
  {
    return false;
  }
}
```

### 신호등 색 필터링

신호등 색 인지에서 그 날 외부에서 들어오는 빛의 광량에 따라 V값이 민감하게 반응을 하여 낮 시간대에 잘 작동하던 HSV 세팅 값이 저녁이 되면 또 다르게 세팅을 해주어야 했다. 민감하게 반응하는 부분을 덜 민감하게 하기 위해서 count를 사용하여 개수를 세다가 일정 수가 넘어가면 멈추도록 설정했다.

```cpp
void LaneKeepingSystem::run()
{
  int lpos, rpos, error, ma_mpos;
  float steering_angle;
  while (ros::ok())
  {
    ros::spinOnce();
    if (frame_.empty() || (bbox_flag_ == false))
    {
      continue;
    }
    if (object_id_ == 0)
    {
      drive_normal("left", 15.0);
      stop_count_ = 0;
    }
    else if (object_id_ == 1)
    {
      drive_normal("right", 15.0);
      stop_count_ = 0;
    }
    else if ((object_id_ == 2) || (object_id_ == 3))
    {
      drive_stop("straight", 6.0, 2.0);
    }
    else if (object_id_ == 4)
    {
      traffic_light_ = traffic_sign_detect(frame_, xmin_, xmax_, ymin_, ymax_)
      if (traffic_light_ == false)
      {
        ++red_count_;
        if (red_count_ > 3)
        {
          drive_stop("straight", 2.0, 0.0);
          green_count_ = 0;
        }
      }
      else
      {
        ++green_count_;
        if (green_count_ > 6)
        {
          drive_normal("go", 0.0);
          red_count_ = 0;
        }
      }
    }
    else
    {
      drive_normal("go", 0.0);
    }
    object_id_ = -1;
  }
}
```

# 정지선 인식 방법

---

### 인식 방법

- HLS 색공간을 이용하여 명도차를 기반으로 OpenCV의 countNonZero함수를 이용하여  ROI의 픽셀수를 count하는 방식을 정지선을 인식하였습니다. 하지만 정지선 역시 실내였음에도 불구하고 조도에 따라 값 세팅을 새로 해주어야 했기에 보다 강건한 방법이 필요하다고 느꼈습니다. 정지선은 횡단보도 표지판, 신호등, 일시정지 표지판 앞에 일정한 간격으로 있는것을 이용하여 검출된 횡단보도, 신호등, 일시정지 표지판의 BoundingBox 사이즈를 이용하여 정지선 앞에서 정지하게 하였습니다.

# 표지판 등 인식 관련

---

- Bounding box 크기로 필터링을 진행하였습니다
- 정지선 검출 대신 bbox 크기만으로 정지를 하다보니 교차로 구간을 지나가다 인식 할 필요가 없는 신호등이 인지가 되지 않아야 하는 경우에도 인지가 될 때가 있어서 bbox의 높이/bbox넓이 로 계산한 후 일정 수준 이상 커지면 필터링을 해주었습니다.
    
    ![KakaoTalk_20230905_203036328.jpg](Object%20Detection%20212b67471a484b2d9fd0e0be829cd663/KakaoTalk_20230905_203036328.jpg)
    

```cpp
void LaneKeepingSystem::bboxCallback(const yolov3_trt_ros::BoundingBoxes& msg)
{
  float traffic_slpe = 0.0F;
  bbox_flag_ = true;
  for (auto& bbox : msg.bounding_boxes)
  {
    xmax_ = bbox.xmax;
    xmin_ = bbox.xmin;
    ymax_ = bbox.ymax;
    ymin_ = bbox.ymin;
    traffic_sign_space_ = (xmax_ - xmin_) * (ymax_ - ymin_);
    traffic_slope = static_cast<float>(ymax_ - ymin_) / static_cast<float>(xmax - xmin)
    if ((bbox.id == 0) || (bbox.id == 1) && (traffic_sing_space_ >= 1500))
    {
      object_id_ = bbox.id;
    }
    else if ((bbox.id == 2) || (bbox.id == 3) && (traffic_sing_space_ >= 3500))
    {
      object_id_ = bbox.id;
    }
    else if ((bbox.id == 4) && (traffic_sign_space_ >= 10000) && (traffic_sign_space_ < 25000) &&
    (traffic_slope < 2.5))
    {
      object_id_ = bbox.id;
    }
  }
}
```

# 결론

### 결과

- 모든 표지판, 신호등 정지선 인식 완료
- 차선 이탈 0회, 감점 0점
- Object Detection Project 대회 1등

### 회고

- 딥러닝에 있어 가장 중요한 부분은 데이터라고 생각하여 많은 양의 데이터 수집에 집중하였다. 또한 단순히 데이터의 많은 수 뿐만 아니라 데이터의 질까지 생각하여 팀원과 함께 라벨링 기준, 검수까지 함께 했던 전략이 유효했던 것 같다.
- 코드로 구현한 것을 하드웨어에서 실제 구동을 하고 코드상에서 예상 했던 것과 다르게 발생한 문제점을 해결하는 부분이 매우 흥미로운 프로젝트였던 것 같다.
- 강의장이라는 실내 환경임에도 불구하고 조도에 따라 색상 처리 부분, 명도 변화가 심하였는데 실외 환경에서 현업에선 이러한 문제들을 딥러닝을 이용한 방법이 아니라면 어떻게 해결하는지 처리 과정이 궁금해졌다.

---