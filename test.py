# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 00:54:19 2021

@author: Made by Kyujin Han
"""

# 모듈 불러오기
from FaceDetector import *
from emotion_detector import *

# 얼굴이 누구인지 recognition을 진행하고 추가적으로 나이와 감정을 분류하는 코드
def main():
    face_detector = FaceDetector()
    face_label = face_detector.face_detector()
    emotion_classification = Emotion(face_label)
    emotion_classification.emotion_check()
    if input("당신의 얼굴에 대한 감정분석을 진행할까요? (y/n)") == "y":
        print()
        emotion_classification.emotion_analysis()
        
    print("프로그램을 종료합니다.")

main()

    