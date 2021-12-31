import tensorflow as tf
import os
import pathlib
import time

import numpy as np
import zipfile

import matplotlib.pyplot as plt
from PIL import Image

import cv2 

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from object_detection_video_function import save_inference, show_inference



# 내 로컬에 설치된 레이블 파일을, 인덱스와 연결시킨다.
PATH_TO_LABELS = 'D:\\TensorFlow\\models\\research\\object_detection\\data\\mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)


# 모델 로드하는 함수.

# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
# 위의 사이트에서 모델을 가져올수있다.

# /20200711/efficientdet_d0_coco17_tpu-32.tar.gz
# http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz

# Download and extract model
def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

MODEL_DATE = '20200711'
MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)

def load_model(model_dir) :
    model_full_dir = model_dir + "/saved_model"

    # Load saved model and build the detection function
    detection_model = tf.saved_model.load(model_full_dir)
    return detection_model

detection_model = load_model(PATH_TO_MODEL_DIR)




# 비디오를 실행하는 코드
cap = cv2.VideoCapture('data/video.mp4')

if cap.isOpened()  == False :
    print('비디오 실행 에러')
else :
    # 프레임의 정보를 가져와 보기!
    # 화면크기를 말하는 것! (width, height)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('data/output3.avi', 
                        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                        20,
                        (frame_width, frame_height))
    # 비디오 캡쳐에서, 이미지를 1장씩 가져온다.
    # 이 1장의 이미지를, 오브젝트 디텍션 한다.
    while cap.isOpened() :
        # 읽어온 이미지를 넘파이(frame)로 가져온다.
        ret, frame = cap.read()

        if ret == True :
            # frame 이 이미지에 대한 넘파이 어레이 이므로
            # 이 frame 을 오브젝트 디텍션 한다.
            # 이 부분은 이미지에서도 비디오에서도 동일하게 반복되기에 함수로 만들었다.
            
            # 동영상을 실시간으로 화면에서 디텍팅하는 것
            # show_inference(detection_model, frame, category_index)
            # 학습용으로, 동영상으로 저장하는 코드로 수정하였다.
            
            start_time = time.time()
            # 동영상을 디텍션 한 후, 파일로 저장하는 함수
            save_inference(detection_model, frame, category_index, out )
            end_time = time.time()
            print('연산에 걸린 시간', str(end_time-start_time))

            if cv2.waitKey(25) & 0xFF == 27:
                break
        else :
            break

    # 비디오 캡쳐 닫고
    cap.release()
    out.release()
    # 윈도우 창 닫고
    cv2.destroyAllWindows()






