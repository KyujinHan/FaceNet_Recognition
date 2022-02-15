# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 00:55:46 2021

@author: Made by Kyujin Han
"""

# import module
from deepface import DeepFace
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import os

# Emotion classification
class Emotion:
    def __init__(self, face):
        self.emotion = DeepFace
        self.path = "./real_time_image"
        self.target = "./real_time_image/"+face
        
    # 검출된 얼굴의 사진 이미지 감정 분석 하기
    def emotion_check(self):
        """
        https://github.com/serengil/deepface/blob/f877591abcca33eff8af06d9a6eaec279a597dc8/deepface/DeepFace.py#L267
        deepface github 자료
        """
        self.result = self.emotion.analyze(img_path=self.target, actions=["age","emotion"], detector_backend = "mtcnn")
        
    # 폴더 안에 있는 이미지 제거
    def remove_real_time_image(self):
        for file in os.listdir(self.path):
            os.remove(os.path.join(self.path,file))
        
    # 자신의 얼굴에 대한 정보를 알고 싶을 때
    def emotion_analysis(self):

        if self.result["dominant_emotion"] == "angry":
            print("당신은 화가나 보이는 군요.", end="// ")
            print("약 ",np.round(self.result["emotion"]["angry"],1),"% 확률",sep="")
            
        elif self.result["dominant_emotion"] == "fear":
            print("당신은 어딘가 두려워 보이는 군요. ", end="// ")
            print("약 ",np.round(self.result["emotion"]["fear"],1),"% 확률",sep="")
            
        elif self.result["dominant_emotion"] == "neutral":
            print("당신의 표정은 무표정과 가깝군요.", end="// ")
            print("약 ",np.round(self.result["emotion"]["neutral"],1),"% 확률",sep="")
            
        elif self.result["dominant_emotion"] == "sad":
            print("당신의 얼굴이 슬퍼 보이는 군요.", end="// ")
            print("약 ",np.round(self.result["emotion"]["sad"],1),"% 확률",sep="")
            
        elif self.result["dominant_emotion"] == "disgust":
            print("당신의 표정에서 역겨움이 보이는 군요.", end="// ")
            print("약 ",np.round(self.result["emotion"]["disgust"],1),"% 확률",sep="")
            
        elif self.result["dominant_emotion"] == "happy":
            print("당신은 행복해 보이는 군요.", end="// ")
            print("약 ",np.round(self.result["emotion"]["happy"],1),"% 확률",sep="")
            
        elif self.result["dominant_emotion"] == "surprise":
            print("당신은 깜짝 놀란 것 같군요.", end="// ")
            print("약 ",np.round(self.result["emotion"]["surprise"],1),"% 확률",sep="")
        
        
        print("\n당신으로 인식한 얼굴을 출력합니다...")
        image = img.imread(self.target)
        plt.imshow(image)
        plt.show()
        
        # 실시간 이미지 제거
        self.remove_real_time_image()
    
            
        
