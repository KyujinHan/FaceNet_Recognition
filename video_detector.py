
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:07:25 2021

@author: Admin
"""

# #############################################
# face detection with mtcnn on live cam feed  #
###############################################
import warnings
warnings.filterwarnings("ignore")
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import cv2
import pandas as pd
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from sklearn.preprocessing import Normalizer, LabelEncoder
#import pickle
from inception_resnet_v1 import *
import time


class FaceDetector:

    def __init__(self):
        self.facenet_model = InceptionResNetV1()
        #self.svm_model = pickle.load(open("D:\\PYTHON_CODE\\Face_Recognition\\SVM_classifier.sav", 'rb'))
        #self.data = np.load('D:\\PYTHON_CODE\\Face_Recognition\\faces_dataset_embeddings.npz')
        # object to the MTCNN detector class
        self.detector = MTCNN()

    def face_mtcnn_extractor(self, frame):
        """Methods takes in frames from video, extracts and returns faces from them"""
        # Use MTCNN to detect faces in each frame of the video
        result = self.detector.detect_faces(frame)
        return result

    def face_localizer(self, person):
        """Method takes the extracted faces and returns the coordinates"""
        # 1. Get the coordinates of the face
        bounding_box = person['box']
        x1, y1 = abs(bounding_box[0]), abs(bounding_box[1])
        width, height = bounding_box[2], bounding_box[3]
        x2, y2 = x1 + width, y1 + height
        return x1, y1, x2, y2, width, height

    def face_preprocessor(self, frame, x1, y1, x2, y2, check="y", check0="n", name=None, i=0, required_size=(160, 160)):
        """Method takes in frame, face coordinates and returns preprocessed image"""
        # 1. extract the face pixels
        face = frame[y1:y2, x1:x2]
        # 2. resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        
        # 신규 멤버이면 사진 저장
        if check=="n" or check0=="y":
            img = Image.fromarray(face_array,"RGB")
            img.save("./lfw_over_10/"+name+"_"+str(i)+".jpg")
        
        # 3. scale pixel values
        face_pixels = face_array.astype('float32')
        # 4. standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # 5. transform face into one sample
        samples = np.expand_dims(face_pixels, axis=0)
        # 6. get face embedding
        self.facenet_model.load_weights("./facenet_weights.h5")
        yhat = self.facenet_model.predict(samples)
        face_embedded = yhat[0]
        # 7. normalize input vectors
        in_encoder = Normalizer(norm='l2')
        X = in_encoder.transform(face_embedded.reshape(1, -1))
        return X
    
    """
    def face_softmax_classifier(self, X):
        #Methods takes in preprocessed images ,classifies and returns predicted Class label and probability
        # predict
        yhat = self.svm_model.predict(X)
        label = yhat[0]
        yhat_prob = self.svm_model.predict_proba(X)
        probability = round(yhat_prob[0][label], 2)
        trainy = self.data['arr_1']
        # predicted label decoder
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        predicted_class_label = out_encoder.inverse_transform(yhat)
        label = predicted_class_label[0]
        return label, str(probability)
    """
    
    def calculate_embedding(self,x):
        embedding_all_128_l2 = np.load("./embedding_all_128_L2.npy")
        df_all_encoder = pd.read_csv("./df_all_encoder_10.csv")
        
        # Calculate
        emd1 = x
        dist_min = 99999 # 거리가 가장 가까울 때 가지는 값의 변수
        point = 0 # 거리가 가장 가깝게 되는 이미지 번호
       
        for j in list(range(len(df_all_encoder))):
            emd2 = embedding_all_128_l2[j]
            emd_dist = self.euclidean(emd1,emd2)
        
            # 가장 거리가 가깝게 되는 이미지
            if dist_min >= emd_dist:
                dist_min = emd_dist
                point = j
        
        # 가장 가까운 거리 (== 인식하는 얼굴 대상) // threshold: 1.242
        if dist_min < 1.1:
            image_name = df_all_encoder.iloc[point,0]
            label = image_name.split("_0")[0]
            
            return label, round(dist_min,3)
        
        # 같은 사람이 없다고 인식 될 때
        elif dist_min >= 1.1:
            label = "None"
            
            return None, 99999
        
    def euclidean(self,x1,x2):
        dist = np.sqrt(np.sum(np.square(x1-x2)))
        return dist
    
    
    """index out of range라는 error 발생"""
    # 새로운 사진 넣기 
    def input_myface(self, check="y"):
        cap = cv2.VideoCapture(0)
        
        if check=="n":
            name = input("당신의 이름을 입력해주세요")
                
            # 10장의 이미지 저장 후 embedding 계산
            df_all_encoder = pd.read_csv("./df_all_encoder_10.csv") # Data frmae 불러오기
            name_list = []
            category = []
            number = df_all_encoder["category"].values[-1]+1 # category 지정
            x_list = np.zeros((10,128))
                
            for i in range(1,11):
                print(i,"번째 사진 저장하는 중...")
                
                __, frame = cap.read()
                # display the frame with label
                cv2.imshow('frame', frame)
                time.sleep(0.5)
                
                result = self.face_mtcnn_extractor(frame)
                x1, y1, x2, y2, width, height = self.face_localizer(result[0]) # 가장 메인이 되는 얼굴
                # 신규 사진 저장 및 embedding 추출
                X = self.face_preprocessor(frame, x1, y1, x2, y2, "n", "n", name, i, required_size=(160, 160)) # Make embedding
                x_list[i-1] = X
                category.append(number) 
                name_list.append(name+"_"+str(i)+".jpg")
                
                
                
            # Embedding 갱신
            embedding_all_128_l2 = np.load("./embedding_all_128_L2.npy") # 기존의 embedding data 불러오기
            embedding_list = np.concatenate((embedding_all_128_l2,x_list),axis=0)
            
            # 데이터 프레임 갱신
            df = pd.DataFrame({"image":name_list,
                               "category":category})
            
            df_all_encoder = pd.concat([df_all_encoder,df],axis=0)
            
            df_all_encoder.to_csv("./df_all_encoder_10.csv", index=False)
            np.save("./embedding_all_128_L2.npy",embedding_list) # 저장
            
    
    # 가입했는데 인식을 못할 때 사진 추가하기
    def check_myface_rewrite(self, check0="n"):
        cap = cv2.VideoCapture(0)
        
        if check0=="y":
            name = input("당신의 이름을 입력해주세요 (기존의 가입했던 이름(ID)으로 적어주세요)")
            
            # 기존의 이미지 라벨 찾기
            df_all_encoder = pd.read_csv("./df_all_encoder_10.csv") # Data frmae 불러오기
            name_df = [names.split("_")[0] for names in df_all_encoder["image"]]
            find_name = np.where(name_df==np.array([name]*len(name_df)))
            
                
            # 10장의 이미지 저장 후 embedding 계산
            name_list = []
            category = []
            number = df_all_encoder.iloc[find_name[0][0],1] # 기존의 정보와 같은 category 지정
            count = df_all_encoder.loc[find_name[0]]
            x_list = np.zeros((10,128))
                
            for i in range(len(count),len(count)+10):
                print(i-len(count),"번째 사진 저장하는 중...")
                
                # display the frame with label
                cv2.imshow('frame', frame)
                time.sleep(0.5)
                
                # 얼굴을 인식 못하거나 2명 이상일 경우 제외
                while True:
                    __, frame = cap.read()
                    result = self.face_mtcnn_extractor(frame)
                    if len(result)==1:
                        break
                    
                x1, y1, x2, y2, width, height = self.face_localizer(result[0]) # 가장 메인이 되는 얼굴
                # 신규 사진 저장 및 embedding 추출
                X = self.face_preprocessor(frame, x1, y1, x2, y2, "y", "y", name, i, required_size=(160, 160)) # Make embedding
                x_list[i-len(count)] = X
                category.append(number) 
                name_list.append(name+"_"+str(i)+".jpg")
                
                
            # Embedding 갱신
            embedding_all_128_l2 = np.load("./embedding_all_128_L2.npy") # 기존의 embedding data 불러오기
            embedding_list = np.concatenate((embedding_all_128_l2,x_list),axis=0)
            
            # 데이터 프레임 갱신
            df = pd.DataFrame({"image":name_list,
                               "category":category})
            
            df_all_encoder = pd.concat([df_all_encoder,df],axis=0)
            
            df_all_encoder.to_csv("./df_all_encoder_10.csv", index=False)
            np.save("./embedding_all_128_L2.npy",embedding_list) # 저장
            
        
    def face_detector(self):
        """Method classifies faces on live cam feed
           Class labels : sai_ram, donald_trump,narendra_modi, virat_koli"""
        # open cv for live cam feed
        # cap = cv2.VideoCapture(0)
        
        """
        처음에 신규가입이 되었는지 확인해본다.
        """
        """
        # 신규가입 확인
        check = input("우리 회사에 가입하신 적이 있나요? (아니라면 n, 맞다면 아무거나 눌러주세요)")
        if check=="n":
            name = input("당신의 이름을 입력해주세요")
                
            # 10장의 이미지 저장 후 embedding 계산
            df_all_encoder = pd.read_csv("./df_all_encoder_10.csv") # Data frmae 불러오기
            name_list = []
            category = []
            number = df_all_encoder["category"].values[-1]+1 # category 지정
            x_list = np.zeros((10,128))
                
            for i in range(1,11):
                print(i,"번째 사진 저장하는 중...")
                
                
                # 얼굴을 인식 못하거나 2명 이상일 경우 제외
                # 1장만
                while True:     
                    __, frame = cap.read()
                    result = self.face_mtcnn_extractor(frame)
                    if len(result)==1:
                        break
                
                x1, y1, x2, y2, width, height = self.face_localizer(result[0]) # 가장 메인이 되는 얼굴
                # 신규 사진 저장 및 embedding 추출
                X = self.face_preprocessor(frame, x1, y1, x2, y2, check, "n", name, i, required_size=(160, 160)) # Make embedding
                x_list[i-1] = X
                category.append(number) 
                name_list.append(name+"_"+str(i)+".jpg")
                
                # 5. Draw a frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 155, 255), 2)
                
                # display the frame with label
                cv2.imshow('frame', frame)
                
                
            # Embedding 갱신
            embedding_all_128_l2 = np.load("./embedding_all_128_L2.npy") # 기존의 embedding data 불러오기
            embedding_list = np.concatenate((embedding_all_128_l2,x_list),axis=0)
            
            # 데이터 프레임 갱신
            df = pd.DataFrame({"image":name_list,
                               "category":category})
            
            df_all_encoder = pd.concat([df_all_encoder,df],axis=0)
            
            df_all_encoder.to_csv("./df_all_encoder_10.csv", index=False)
            np.save("./embedding_all_128_L2.npy",embedding_list) # 저장
        
            
        k = 1 # 연속 카메라
        u = 1 # 시행횟수
        """
        cap = cv2.VideoCapture(0)
        # 회원 가입이 됬거나 or 기존 회원이면 // 5번 face recognition
        while True:
            # Capture frame-by-frame
            __, frame = cap.read()
            # 1. Extract faces from frames
            result = self.face_mtcnn_extractor(frame)
            if result:
                for person in result:
                    # 2. Localize the face in the frame
                    x1, y1, x2, y2, width, height = self.face_localizer(person)
                    # 3. Proprocess the images for prediction
                    X = self.face_preprocessor(frame, x1, y1, x2, y2, required_size=(160, 160))
                    # 4. Predict class label and its probability
                    
                    label, dist = self.calculate_embedding(X)
                    print(" Person : {} , Dist : {}".format(label, dist))
                    
                    # 5. Draw a frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 155, 255), 2)
                    # 6. Add the detected class label to the frame
                
                    cv2.putText(frame, label+str(dist), (x1, height),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                                lineType=cv2.LINE_AA)
                    #break # 1장만 인식 되도록
            
            # display the frame with label
            cv2.imshow('frame', frame)
            # break on keybord interuption with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            """
            # 연속적으로 촬영하는 영상을 보여주기 위한 if 문
            if k%5==0:
            
                # 내가 맞는지 확인 - 종료
                isityour = input("당신의 정보가 맞습니까? (맞다면 y, 아니라면 아무거나 입력해주세요.)")
                if isityour=="y":
                    print("당신의 정보를 불러옵니다...")
                    print("당신의 이름은",label.split("_")[0])
                    print("환영합니다 ^^")
                    
                    break
                
                # 틀린 사진 빼고 recognition
                else:
                    next
                
                # 다시 사진을 넣을 것인지 물어보기 (face recognition 실패하는 경우)
                if u==1:
                    check0 = input("사진을 다시 넣으시겠습니까? (y/n)")
                    
                    if check0=="y":
                        name = input("당신의 이름을 입력해주세요 (기존의 가입했던 이름(ID)으로 적어주세요)")
                        
                        # 기존의 이미지 라벨 찾기
                        df_all_encoder = pd.read_csv("./df_all_encoder_10.csv") # Data frmae 불러오기
                        name_df = [names.split("_")[0] for names in df_all_encoder["image"]]
                        find_name = np.where(name_df==np.array([name]*len(name_df)))
                        
                        
                        # 10장의 이미지 저장 후 embedding 계산
                        name_list = []
                        category = []
                        number = df_all_encoder.iloc[find_name[0][0],1] # 기존의 정보와 같은 category 지정
                        count = df_all_encoder.loc[find_name[0]]
                        x_list = np.zeros((10,128))
                        
                        for i in range(len(count),len(count)+10):
                            print(i,"번째 사진 저장하는 중...")

                            
                            # 1장만
                            while True:     
                                __, frame = cap.read()
                                result = self.face_mtcnn_extractor(frame)
                                if len(result)==1:
                                    break
                                
                            x1, y1, x2, y2, width, height = self.face_localizer(result[0]) # 가장 메인이 되는 얼굴
                            # 신규 사진 저장 및 embedding 추출
                            X = self.face_preprocessor(frame, x1, y1, x2, y2, "y", "y", name, i, required_size=(160, 160)) # Make embedding
                            x_list[i-len(count)] = X
                            category.append(number) 
                            name_list.append(name+"_"+str(i)+".jpg")
                            
                            # 5. Draw a frame
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 155, 255), 2)
                            
                            cv2.putText(frame, name+"_"+str(i)+".jpg", (x1, height),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                                lineType=cv2.LINE_AA)
                
                            # display the frame with label
                            cv2.imshow('frame', frame)
                            
                            
                        # Embedding 갱신
                        embedding_all_128_l2 = np.load("./embedding_all_128_L2.npy") # 기존의 embedding data 불러오기
                        embedding_list = np.concatenate((embedding_all_128_l2,x_list),axis=0)
                        
                        # 데이터 프레임 갱신
                        df = pd.DataFrame({"image":name_list,
                                           "category":category})
                        
                        df_all_encoder = pd.concat([df_all_encoder,df],axis=0)
                        
                        df_all_encoder.to_csv("./df_all_encoder_10.csv", index=False)
                        np.save("./embedding_all_128_L2.npy",embedding_list) # 저장
            
                u += 1        
            k += 1
        """
            
        # When everything's done, release capture
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    facedetector = FaceDetector()
    facedetector.face_detector()