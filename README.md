# 수신호 인식과 표정인식을 이용한 위험 상황 인지 알고리즘

논문 : [수신호 인식과 표정인식을 이용한 위험 상황 인지 알고리즘](https://github.com/user-attachments/files/18320088/IPIU2022_Paper.pdf)

### 파일 구조
 - data/: 손동작별(fist, hand out, help, nothing) 손바닥 좌표가 저장되어 있 디렉토리
 - face_data/: 얼굴 표정(angry, happy, neutral, sad, surprise) 관련 파일들이 train dataset과 validation dataset으로 저장된 디렉토리
 - model/: 손동작을 예측하는 LSTM 모델의 가중치 파일이 담겨있는 디렉토리
 - Classification_vgg16.py: 얼굴 표정 데이터를 전처리하고, VGG16을 이용하여 얼굴 표정 데이터를 학습시키는 파일
 - Emotion_vgg16.h5: Classification_vgg16.py 여기서 학습 한 모델의 가중치가 저장되어 있는 파
 - Facial_Expressions_Recog.py: VGG16이 얼굴 표정 인식을 실시간으로 잘 하는지 OpenCV를 통해 웹캠으로 확인하는 파일
 - Gesture_Classification.py: LSTM이 손동작 예측을 실시간으로 잘 하는지 OpenCV를 통해 웹캠으로 확인하는 파일
 - data_collect.ipynb: 손동작 데이터를 OpenCV를 통한 웹캠으로 수집을 위한 파일
 - haarcascade_frontalface_default.xml: 얼굴 검출을 위한 Haar Cascade 분류기 파일
 - main.py: 메인 실행 파일
 - train_model.ipynb: LSTM을 이용하여 손동작을 예측하는 모델 학습을 위한 파일


### Algorithm Flow

<img width="374" alt="algorithmflow" src="https://github.com/user-attachments/assets/6215ad6c-e0d8-4de8-b3a3-7f6f245d6970" />

### Gesture dataset

![datacollection](https://github.com/user-attachments/assets/c4cbacd5-032b-48b4-bca7-8901911d0198)

 - MediaPipe을 이용하여 사진과 같이 손바닥의 좌표를 nparray형태로 저장

### Gesture dataset 수집 방법

<img width="343" alt="datacollectioncam" src="https://github.com/user-attachments/assets/6d1cd9c6-6609-4cb1-8745-9ebe15bcc504" />

 - 데이터 수집 방법은 웹캠을 통해 30초마다 1번씩 지정한 횟수만큼 데이터를 수집

### Gesture Model

<img width="524" alt="스크린샷 2025-01-07 오전 1 57 51" src="https://github.com/user-attachments/assets/6c52adbc-777d-45ea-a8c3-cc76a57b6984" />

### Gesture Model Result
![gestureres](https://github.com/user-attachments/assets/083083b6-01a1-450b-8234-af5096d90e89)

 - Early Stopping을 통해서 epoch 20에서 모델 가중치 데이터를 파일로 저장
 - Train Acc : 83.33% / Test Acc : 91.67%

### Emotion Model (VGG 16)

참고 논문 : [VGG16 Paper](https://arxiv.org/pdf/1409.1556)

![vgg16](https://github.com/user-attachments/assets/7ef877ef-f514-4e9f-98bb-8430327e15cc)

 - 학습 데이터가 부족하여 추가적인 data augmentation을 진행
 - 논문에서는 RGB이미지를 사용하지만 여기서는 GRAY스케일을 사용하기 때문에 input data의 dimention을 수정
 - VGG16 논문에서 B Network 사용
 - Overfitting을 줄이기 위해 추가적으로 layer 사이에 dropout을 추가
 - Early Stopping 적용
 - Train Acc : 65.70% / Test Acc : 74.13%


#### 자세한 내용은 해당 [발표자료](https://github.com/user-attachments/files/18322623/IPIU2022_._pdf.pdf)를 참고하세요.
