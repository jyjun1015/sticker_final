################# Modules #######################################################
import sys
import datetime             # for getting date
import cv2                  # for using OpenCV4.5 and CUDNN
import numpy as np          # for getting maximum value
import os                   # for getting system information
import time                 # for using time.sleep function
import copy                 # for using CSI_Camera module
from csi_camera import CSI_Camera # for using pi-camera in Jetson nano
##################################################################################

# Find AO(Always On) sensor's value
def checkHeadRatio(raw, stick) :    # 라이터 헤드를 찾기 위한 좌표 추정 함수
    return int((stick-raw) * (5/105))    #헤드 간 간격/몸통 길이

def checkRawRatio(candidate) :  # 라이터 헤드 아랫부분 기준을 찾기 위한 좌표 추정 함수
    return int(candidate * (274/490))   #헤드 아랫 부분/라이터 고정대 여기 사이즈 바꿔야 함

def checkStickRatio(raw) : # 스티커 높이 파악 -> 해상도마다 비율을 다르게 설정해줘야 함. 현재 1920 * 1080
    return int(raw * (490/237))
    # 스티커 상단 부분 (바코드)

##################################################################################


def initialize_camera():
    camera = CSI_Camera()
    camera.create_gstreamer_pipeline(
        sensor_id = 0,
        sensor_mode = 2,
        framerate = 30,
        flip_method = 0,
        display_height = 720,
        display_width = 1280
    )
    camera.open(camera.gstreamer_pipeline)
    camera.start()
   

def ShowMainMenu():   
    """
    Show Main Options.
    """
    print()
    print("[메인 옵션 선택]")
    for idx, val in enumerate(options.keys()):
        print(idx+1, val)
    global check_main_menu 
    print()
    check_main_menu = int(input("원하는 기능의 번호를 입력하세요 : "))

def ShowCameraMenu():
    """
    Show Camera Options.
    """
    print()
    print("[카메라 옵션 선택]")
    for idx, val in enumerate(options['카메라 실행']):
        print(idx+1, val)
    global check_camera_menu
    print()
    check_camera_menu = int(input("원하는 기능의 번호를 입력하세요 : "))
 
def YoLo():
    """
    Args:
        
    Returns:
        
    Raises:
    
    Note:
        This function does not perform any special functions, 
        but only performs to show images from the camera to the GUI in real time.
        And it visually helps user can easily control sliders and buttons.
    """
    global position
    position = []
    
    
    blob = cv2.dnn.blobFromImage(img, 0.00392, (448, 448), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    confidences = []
    boxes = []
    boxes_temp = []
    boxes_low = -1
    for detection in outs[1]:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.7 and class_id == 0 :
            center_x = int(detection[0] * DISPLAY_WIDTH)
            center_y = int(detection[1] * DISPLAY_HEIGHT)
            w = int(detection[2] * DISPLAY_WIDTH)
            h = int(detection[3] * DISPLAY_HEIGHT)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes_temp.append([x, y, w, h])
            # +20의 경우 바운딩 박스 사이 간격을 주기 위함 
            boxes_low = max(boxes_low, y+h+20)
            confidences.append(float(confidence))
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    indexes = cv2.dnn.NMSBoxes(boxes_temp, confidences, 0.3, 0.2)
    boxes = []
    for i in range(len(boxes_temp)):
        if i in indexes:
            x, y, w, h = boxes_temp[i]
            boxes.append([x, y, w, h])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

    temp = boxes_low 
    temp2 = checkStickRatio(temp)

    if temp2 > 0 :
        stick = temp2
    if temp > 0 :
        raw = temp
        
    boxes.sort(key=lambda x : x[0])
    
    # 헤드와 헤드 사이
    between = checkHeadRatio(raw, stick)

    error_minVal = 1
    start = boxes[0][0]-between if boxes[0][0]-between >= 0 else 0
    end = boxes[-1][0]+boxes[-1][2] + between if boxes[-1][0]+boxes[-1][2] + between < 720 else 719

    for index in boxes :

        start = index[0] if index[0]-between >= 0 else 0
        # start = index[0]-between if index[0]-between >= 0 else 0

        end = index[0]+index[2] 
        # resize_width, resize_height = checkStickerRatio(raw, stick)
        resize_width, resize_height = checkStickerRatio(raw, stick)
    
        position.append([start, raw, end, stick])
        # 바디 사각형 그리기 -> 카메라와 이루는 각이 수평이 아니라면, 오차 발생
        cv2.rectangle(img, (start,raw), (end, stick), (255, 0, 0), 2, cv2.LINE_8)
        
    return img, position

def TemplateMatching():
    """
    Args:
        
    Returns:
        
    Raises:
    
    Note:
    """
    for idx, pos in enumerate(position):
        # 이미지를 잘라 개별 스티커 검출 
        cut_img = img[pos[1]:pos[3], pos[0]:pos[2]]
        # Blur 처리
        
        cut_img = cv2.medianBlur(cut_img,3)
        
        # 이미지 조정 
        cut_img = cut_img[5:-4,10:-3]
        
        cv2.imwrite("temp.jpg", cut_img)
        time.sleep(3)
        
        resul = []
        try :
            result = cv2.matchTemplate(cut_img, sticker, cv2.TM_SQDIFF_NORMED)
        except :
            print("템플릿에러")
            continue;
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
        x, y = minLoc
        h, w, c = sticker.shape
        print(minVal)
        resul.append([pos[0]+x, pos[1]+y, w, h, minVal])
        resul.sort(key = lambda x : x[4])
        if resul[0][4] <= 0.1 :
    
            results.append(resul[0])
            answer[idx] = 1
    
    print(answer)
    print()
    return results, answer

def FindErrorSticker(array):
    error_index = []
    cnt = 0
    for idx,ans in enumerate(array):
        if ans == 0:
            cnt += 1
            error_index.append(idx+1)
    if cnt == 0 :
        return 
    return error_index

if __name__ == '__main__':

############## Resolution Static Variables ##########
    DISPLAY_WIDTH   = 1280  # Display frame's width
    DISPLAY_HEIGHT  = 720   # Display frame's height
    raw = 274 # 라이터 고정대 길이 
    stick = 490

############## Pi-Camera for jetson nano ################ 
    #initialize_camera()
    camera = CSI_Camera()
    camera.create_gstreamer_pipeline(
        sensor_id = 0,
        sensor_mode = 0,
        framerate = 30,
        flip_method = 2,
        display_height = DISPLAY_HEIGHT,
        display_width = DISPLAY_WIDTH
    )
    camera.open(camera.gstreamer_pipeline)
    camera.start()
############## YoLo Static Variables ################
    net = cv2.dnn.readNet("yolov4-tiny_4000.weights", "yolov4-tiny.cfg")    
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

############## Template Static Variables ############
    sticker = cv2.imread("./new_template/"+str(15)+'.jpg')
############## lighter Static Variables #############
    lighter_number = 10 
    count_set = 1
    max_frame_number = 10
    position = []

############## Options Static Variables #############
    options = {"카메라 실행" : ["카메라 위치 설정", "솔루션 실행", "뒤로가기"],
           "GUI 실행" : [],
          "종료" : []}

    main_state = True
    while main_state:
        ShowMainMenu()
        if check_main_menu == 1: 
            camera_state = True
            while camera_state: 
                ShowCameraMenu()
                if check_camera_menu == 1: # 카메라 위치 설정 ON 
                    camera_position_state = True
                    print("종료하기 : Q 버튼")
                    while camera_position_state:
                        _, img = camera.read()
                        try :
                            #-----라이터 헤드를 기준으로 바디를 추정-----#
                            YoLo()
                            cv2.imshow('frame', img)
                            if cv2.waitKey(100) & 0xFF == ord('q') and len(position) == lighter_number:
                                camera_position_state = False
                        except :
                            continue
                
                    cv2.destroyAllWindows()
                    print("완료")
                elif check_camera_menu == 2:
                    print("종료하기 : Q 버튼")
                    print(position)
                    solution_state = True
                    while solution_state :
                        ret, img = camera.read()
                        cv2.imshow('frame', img) 
                        if cv2.waitKey(81) & 0xFF == ord('w'):
                            results = []
                            answer = [0 for _ in range(lighter_number)]
                            today = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            HowMany = max_frame_number
                            count_set += 1
                            while HowMany :
                                _, img = camera.read()
                                cv2.imshow('frame', img)
                                HowMany -= 1
                                TemplateMatching()
                                if 0 not in answer :
                                    print(today, count_set,"번 세트 정상")
                                    for idx,val in enumerate(results):
                                        cv2.rectangle(img, (val[0], val[1]), (val[0]+val[2], val[1]+val[3]), (255, 255, 0), 2, cv2.LINE_8)
                                    cv2.imshow('frame', img)
                                    HowMany = 0 # False
                            if 0 in answer : 
                                print(today, count_set,"번 세트",FindErrorSticker(answer),"번 스티커 불량")
                            cv2.imshow('frame', img)
                        elif cv2.waitKey(31) & 0xFF == ord('q'):
                        
                            cv2.destroyAllWindows()
                            solution_state = False

                    print("완료")

                elif check_camera_menu == 3: # 뒤로가기 -> 상태변환
                    camera_state = False

        elif check_main_menu == 2:
            print()
            print("준비중입니다.....")
        elif check_main_menu == 3:
            main_state = False
        else:
            print()
            print("다시 입력하세요")

print()
print("Good Bye")
camera.stop()
camera.release()
sys.exit(0)
