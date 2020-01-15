#!/usr/bin/env python
# coding: utf-8
from numpy import array
from numpy import asarray
from numpy import expand_dims
import os
from PIL import Image, ImageFont, ImageDraw
import datetime
import time
import detect_embedding
import pandas as pd
import pickle
from tqdm import tqdm_notebook
import time
import cv2

class CSV_write : # pandas 를 이용하여, csv를 저장합니다.
    def __init__(self, video_file, PATH) :
        self.PATH = PATH
        self.video_file_path = PATH+'video/'+video_file 
        self.video_file = video_file
        self.vod_path = os.path.realpath(self.video_file_path)
        self.result_df = pd.DataFrame(columns=['VOD_PATH', "LABEL", "TIMESTAMP"])

    def concat_df(self, name, time) :
        df = pd.DataFrame([{"VOD_PATH" : self.vod_path, "LABEL" : name, "TIMESTAMP": time}])
        df = df[['VOD_PATH', 'LABEL', 'TIMESTAMP']]
        self.result_df = pd.concat([self.result_df, df])
        
    def write(self) :
        self.result_df.to_csv(self.PATH+'csv/'+self.video_file+'.csv', header = True, index = False, encoding = 'UTF-8')
        print("csv 저장 완료")

class VIDEO_face_recognition : 
    def __init__(self, video_file, save_file = 'save', percentage = 88, frame_per_detect = 30):
        self.PATH = '/Users/jeonminwoo/Downloads/facenet-mtcnn-svm/'
        self.font = ImageFont.truetype(self.PATH+'NanumBarunGothic.ttf',20)
        self.video_file = video_file
        self.percentage = percentage # svm 분류 확률
        self.frame_per_detect = frame_per_detect # 몇 프레임당 detect 을 할건지
        self.CSV = CSV_write(self.video_file, self.PATH)
        self.save_file = save_file
        try:
            with open(self.PATH+'svm_test.pkl', 'rb') as handle:
                face_model = pickle.load(handle)
        except:
            print('svm 모델 불러오기 실패')
        self.model = face_model.get('svm')
        self.in_encoder = face_model.get('in_encoder')
        self.out_encoder = face_model.get('out_encoder')
        print('VIDEO_face_recognition class load ok')

        self.video_read_write()

    def second_to_time(self, second) : # 초를 문자열 hh:mm:ss 형식으로 변환해줍니다.
    
        mydelta = datetime.timedelta(seconds = second)
        video_time = datetime.datetime.min + mydelta
        h, m, s = video_time.hour, video_time.minute, video_time.second
        
        time = '{hour}:{minute}:{second}'
        time = time.format(hour = "%02d" % h, minute = "%02d" % m, second = "%02d" % s)
        
        return time

    def video_read_write(self):

        if os.path.isfile(self.PATH+'video/'+self.video_file) == False:
            print("파일이 없습니다.")
            return False
        
        #opnecv 를 이용하여 비디오파일을 읽습니다.
        cap = cv2.VideoCapture(self.PATH+'video/'+self.video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_second = total_frame / fps
        time = self.second_to_time(total_second)

        print("동영상 길이:", time)
        print("%d frame 당 한번씩 detect 합니다." % self.frame_per_detect)
        print("정상 감지라고 판단하는 svm의 확률은 %d 입니다." % self.percentage)
  

        width = int(cap.get(3))
        height = int(cap.get(4))

        # 새로 저장할 비디오 세팅
        #fcc = cv2.VideoWriter_fourcc('D','I', 'V', 'X')
        fcc = cv2.VideoWriter_fourcc('m','p','4','v')
        out = cv2.VideoWriter(self.PATH+self.save_file+'.mov', fcc, fps, (width, height))
        total_frame = int(total_frame)
        total = tqdm_notebook(range(total_frame))

        for tqdm in total:
            ret, image = cap.read()

            if ret:

                if(int(cap.get(1)%self.frame_per_detect == 0 or cap.get(1) == 1.0)) : # 시작시 한 번 실행, 그 후 %d 프레임마다 얼굴 추출, 비교
                    # CSV 시간 저장을 위해 시간 계산
                    frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    frame_num = int(frame_num)
                    video_second = frame_num / fps
                    time = self.second_to_time(video_second)

                    image, name_list, class_probability_list, bound_box_start, bound_box_end, crop_face_list, status = self.process_image(image)
                    
                    if status :
                        #for name in name_list:
                        #    self.CSV.concat_df(name, time)
                        for i in range(0, len(name_list)):
                            self.CSV.concat_df(name_list[i], time)
                            #crop_face_list[i] = cv2.cvtColor(crop_face_list[i], cv2.COLOR_BGR2RGB)
                            cv2.imwrite(self.PATH+'/crop_img/'+name_list[i]+'%s.jpg'%time,crop_face_list[i])

                box_num = 0
                if status and box_num <3:     
                    for i in range(0, len(name_list)):
                        cv2.rectangle(image, bound_box_start[i], bound_box_end[i], (0,155,255), 2)
                        img_pil = Image.fromarray(image)
                        draw = ImageDraw.Draw(img_pil)
                        draw.text((bound_box_start[i][0], bound_box_end[i][1]), name_list[i]+ ' %.1f' % class_probability_list[i], font=self.font, fill=(0,255,255,0))
                        image = array(img_pil)
                        box_num += 1
                cv2.imshow("test",image)
                out.write(image)

                if(cv2.waitKey(30) & 0xFF == ord('q')):
                    break
            else:
                break

        out.release()
        cap.release() 
        self.CSV.write()
        return True
    

    def process_image(self, image):

        pixels = asarray(image)
        embedding_list, bound_box_list, crop_face_list, status = FA.extract_face_get_embedding(pixels)

        # 에러 혹은 얼굴이 없을 경우 status 가 False 입니다. 다음 사진(frame)으로 넘어갑니다.
        if status == False:
            return pixels, False, False, False, False, False, False

        name_list = []
        class_probability_list = []
        bound_box_start = []
        bound_box_end = []
        crop_face_save = []
        for i in range(0,len(embedding_list)) :
            
            newTrainX = []
            newTrainX.append(embedding_list[i])
            newTrainX = asarray(newTrainX)
            newTrainX = self.in_encoder.transform(newTrainX)
            face_emb = newTrainX[0]

            # prediction for the face
            samples = expand_dims(face_emb, axis=0)
            yhat_class = self.model.predict(samples)
            yhat_prob = self.model.predict_proba(samples)

            # get name
            class_index = yhat_class[0]
            class_probability = yhat_prob[0, class_index] * 100
            predict_names = self.out_encoder.inverse_transform(yhat_class)
            
            if predict_names[0] == 'other':
                continue
            
            if class_probability > self.percentage: # svm 확률이 percentage 이상인 경우에만 맞게 분류한 것으로 간주합니다.
                
                if predict_names[0] in name_list:  
                    #예) 얼굴1 문재인80, 얼굴2 문재인 90일 경우 얼굴1 확률과 박스를 확률이 더 높은 얼굴2로 바꿉니다.
                    #기존 값의 index 확인
                    index = name_list.index(predict_names[0])

                    if class_probability < class_probability_list[index]: # 기존 확률이 더 높다면 coninue
                        continue
                    else:
                        class_probability_list[index] = class_probability # 확률 change

                        # bound box 바꿔주기
                        temp = bound_box_list[i][0], bound_box_list[i][1]
                        temp2 = bound_box_list[i][2], bound_box_list[i][3] 

                        bound_box_start[index] = temp
                        bound_box_end[index] = temp2
                        
                        crop_face_save[index] = crop_face_list[i]

                else:

                    name_list.append(predict_names[0])
                    class_probability_list.append(class_probability)
    
                    temp = bound_box_list[i][0], bound_box_list[i][1]
                    temp2 = bound_box_list[i][2], bound_box_list[i][3]

                    bound_box_start.append(temp)
                    bound_box_end.append(temp2)
                    crop_face_save.append(crop_face_list[i])
            else:
                continue
        return pixels, name_list, class_probability_list, bound_box_start, bound_box_end, crop_face_save, True

if __name__ == '__main__':
    FA = detect_embedding.FaceAnalysis()
# 필수 argument : video_file
# default 값이 있는 argument : save_file (저장할 파일 이름), percentage (svm 확률), frame_per_detect (몇 프레임마다 detect 를 할건지)

    startTime = time.time()
    video_face_recognition = VIDEO_face_recognition(video_file='mun_kim_xi_peng-test.mp4', save_file= '테스트.mp4', frame_per_detect=100)
    endTime = time.time() - startTime
    print("프로그램 실행시간 :", endTime)

