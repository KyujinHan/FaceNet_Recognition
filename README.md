# FaceNet을 이용한 안면인식 모델 구축 프로젝트
# 정보
github 작성자: 한규진   
팀원: 한규진, 송한별, 오승아    
기간: 2021년도 상반기에 진행하여 마무리  
학교: 한동대학교  
  
   
언어: Python  
프레임워크: Tensorflow 2.x version  



# 역할
한규진: 데이터 전처리 및 모델링  
송한별: 데이터 전처리 및 모델링  
오승아: 데이터 전처리 및 Management  


# 설명
원소프트다임 스타트업 회사와 컨택을 해서 간단하게 구현까지 만들어본 안면인식 프로그램.


# 파일 설명
0. 파일에 실행하기 앞서, 빈 폴더 real_time_image 를 만들어야 한다!!  
0-1. real_time_image의 역할은 안면인식을 할 때 실시간으로 찍히는 사진들이 임시로 저장되었다가 안면인식이 완료되었으면 삭제되는 공간이다. 또한 수많은 사진 중에 어떤 사진이 가장 높은 유사도를 보이는지 판단할 때도 필요하다.  
1. lfw_over_10: lfw 데이터셋을 각 사람들이 최소 10장이상 갖고 있는 label에 대해서만 추출  
2. test.py: 프로젝트 스크립트 파일 (전체적인 파일)  
3. FaceDetector.py: 회원가입(임의) 절차와 사람들의 사진을 찍고 embedding값을 저장하는 파일  
4. emotion_detector.py: 감정분석 파일  
5. df_all_encoder_10.csv: labeling 되어있는 csv 파일. 사람들이 새로 입력되거나, 아니면 학습률을 높이기 위해 사진을 추가할 때 csv에 추가된다.  
6. embedding_all_128_L2.npy: numpy 파일로 128차원의 임베딩 값을 가지고 있는 파일이다. L2는 Normalization을 의미한다.  
7. dataframe_embedding.py: numpy 파일과 .csv파일 그리고 lfw_over_10 폴더 안에는 전부 동일한 수치로 데이터(embeddig or 사진)가 들어있다. 해당 파일로 제대로 임베딩 값과 사진, 그리고 데이터가 저장되었는지 확인한다.  
8. video_detector: cv2를 이용해 화면으로 사진을 받아드리는데 참고한 .py파일. (출처는 기억이 안나고, github에서 찾았음)   
9. inception_resnet_v1.py & facenet_weights.h5: FaceNet을 Pre-trained model을 이용하였기에 사용한 파일들.  




# 모델 설명
우리는 FaceNet을 이용했다. 
FaceNet의 Embedding을 "유클리드 거리"계산을 통해 사람과 사람간의 유사도를 계산했다.  
카메라 구현 기능은 cv2를 활용하였고, 해당 코드는 github에서 참고하여 만들었다.  
  
또한 아직 DB를 접해보지 못했던 시기라 임의적으로 df_all_encoder.csv 파일을 만들어서 user의 이름과 label로 사진을 관리했고, FaceDetector.py을 통해 이미 labeling이 되어있는 사람인지 아닌지   확인을 하고, labeling이 안 되어있다면, 신규회원처럼 카메라로 사진의 이미지를 받아와 embedding을 만들고 기존의 데이터셋과 비교하어 사람을 매칭시킨다. (회원가입과 최대한 비슷하게 구현하기 위해 노력)  
+) 여기서 더 나아가 DeepFace의 감정분석을 간단히 적용을 하여서, 사람이 누구이고, 그 사람의 감정이 어떠한지 알려주는 모델을 붙였다.  


Embedding을 이용하지 않고 bottleneck-feature를 붙여 만든 Tranfer-learning 모델은 학습을 진행할 때 케이스를 총 4개로 나누어서 lfw 데이터셋 안에서 각 인물이 사진을 4장 5장, 10장, 20장 이상 가지고 있을 때로 구분지어서 학습을 하였다.



+) FaceNet을 이용해서 밑에 Fc layer를 첨가하여 bottleneck parameter를 update를 하는 방법도 해보았지만, 사용한 데이터가 적어서 그런지 단순히 FaceNet만 이용하였을 때 보다 성능이 낮았다.  


# 우리가 사용한 데이터
우리는 이미지 공공데이터인 "lfw(Labeled Faces in the Wild Home)"를 이용하였고 lfw 데이터를 모두 이용하지 않았다.  
일단 http://vis-www.cs.umass.edu/lfw/ 홈페이지를 참고하여 잘못 라벨링 된 데이터를 모두 수정 하였다.  



# FaceNet 논문 발표 자료
https://docs.google.com/presentation/d/1BWVNAI2qOzjwDZWSgNjEn8lnxPGBxg8s4E3W8aBsASg/edit   
- FaceNet, 그리고 Face Detection 기술인 MTCNN과 FaceNet에 중요한 layer인 Inception(GoogleNet)에 대해 공부하고 발표한 자료이다.  
+) 이 발표를 통해 논문의 중요성을 알았고, 이후 BERT, GAN, VIT를 공부할 때 논문을 무조건 찾아보고 읽게 되었다.   



# 프로젝트 마무리를 위한 최종 발표
https://docs.google.com/presentation/d/1G5pnNO2YNUXhN613R957iZPUQCZAFusI/edit#slide=id.g10678d5f7d6_0_141   



# 기타 TMI
+) 또한 각 사람당 이미지의 기준을 높이면 높일수록 성능이 좋아졌지만, 데이터가 너무나 적어지는 상황도 발생하여서 매우 아쉬웠다.    
+) 한국인을 대상으로 했을 때 성능도 높이기 위해 K-face 데이터를 신청했지만 거절당해서 아쉬운 상황이 발생했었다.   
+) Transfer-learning을 접목한 FaceNet 모델은 해당 Github 저장소에 올리지 않았다. 추후에 기회가 된다면 정리해서 올릴 예정이다.  
+) 처음에는 ResNet을 이용하여 간단한 테스트를 진행하였다.  
+) 우리는 false_image라는 list를 만들어서 만약 안면인식을 진행했을 때 잘못 예측되었다고 사용자가 판단한 이미지라면, 다시 그 이미지를 사용하지 않도록 만들었다.  
+) 사실 Sota Model에서 lfw분야에서 1등을 차지하고 있는 모델은 VarGNet이였다. (2021년 상반기 기준) 그래서 해당 모델을 사용하고 싶었으나 해당 모델은 mxnet 프레임워크로만 구현되어 있어서 사용하기에 어려움을 겪었다.  


