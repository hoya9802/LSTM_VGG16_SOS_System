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

