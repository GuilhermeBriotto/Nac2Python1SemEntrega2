import cv2
import imutils
import numpy as np

def resize(img,new_width=500):
    height,width,_ = img.shape
    ratio = height/width
    new_height = int(ratio*new_width)
    return cv2.resize(img,(new_width,new_height))


# Cria duas variaveis globais
clicks = 0      # conta a quantidade de clicks dada
##coordinates = []

def mouse_click(event, x, y, flags, userdata):

    global clicks
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if clicks >= 3:
            clicks = 1
            print(clicks)
        else:
            clicks += 1
            print(clicks)
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        clicks = 0
        print(clicks)
    

##cv2.VideoCapture("videoFiltro.mp4")
##cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap = cv2.VideoCapture("videoFiltro.mp4")

image = cv2.imread('img/hat.png', cv2.IMREAD_UNCHANGED)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

cv2.namedWindow('Frame')

cv2.setMouseCallback('Frame', mouse_click)
while True:
    
    ret, frame = cap.read()
    if ret == False: break
    
    if clicks == 1:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=640)

        faces = face_cascade.detectMultiScale(frame, 1.3, 5)

        for (x, y, w, h) in faces:

            resized_image = imutils.resize(image, width=w)
            filas_image = resized_image.shape[0]
            col_image = w

            porcao_alto = filas_image // 4

            dif = 0

            if y + porcao_alto - filas_image >= 0:
                n_frame = frame[y + porcao_alto - filas_image : y + porcao_alto, x : x + col_image]
            else:
                dif = abs(y + porcao_alto - filas_image) 
               
                n_frame = frame[0 : y + porcao_alto, x : x + col_image]

            mask = resized_image[:, :, 3]
            mask_inv = cv2.bitwise_not(mask)
            
            bg_black = cv2.bitwise_and(resized_image, resized_image, mask=mask)
            bg_black = bg_black[dif:, :, 0:3]
            bg_frame = cv2.bitwise_and(n_frame, n_frame, mask=mask_inv[dif:,:])

            result = cv2.add(bg_black, bg_frame)
            if y + porcao_alto - filas_image >= 0:
                frame[y + porcao_alto - filas_image : y + porcao_alto, x : x + col_image] = result
            else:
                frame[0 : y + porcao_alto, x : x + col_image] = result
    elif clicks == 2:
        ret ,frame = cap.read()
        frame = resize(frame)
        detections = face_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=6)

        for face in detections:
            x,y,w,h = face

            frame[y:y+h,x:x+w] = cv2.GaussianBlur(frame[y:y+h,x:x+w],(15,15),cv2.BORDER_DEFAULT)
        
    elif clicks == 3:
        ret ,frame = cap.read()
        frame = resize(frame)
        detections = face_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=6)

        for face in detections:
            x,y,w,h = face
        
            kernel = np.array([[-1, 0, -1], 
                   [-1, 11, -1], 
                   [-1, -1, -1]])
            # Realiza o produto de convolução
            frame[y:y+h,x:x+w] = cv2.filter2D(frame[y:y+h,x:x+w],-1,kernel)

    elif clicks == 0:
        ret, frame = cap.read()
    
    
    cv2.imshow('Frame', frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()