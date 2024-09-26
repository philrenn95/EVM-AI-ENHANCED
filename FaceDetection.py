import cv2

def FaceDetection(frames,
                  #masks,
                  ):

    video_croped = []
    #mask_croped = []

    frontal_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

    for index, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_front = frontal_face_cascade.detectMultiScale(gray, 1.1, 4)
        faces_profile = profile_face_cascade.detectMultiScale(gray, 1.1, 4)

        width = frame.shape[0]
        heigth = frame.shape[1]

        if len(faces_front) == 1:
            x, y, w, h = faces_front[0]
            frame = frame[y:y+h, x:x+w]
            #mask = mask[y:y+h, x:x+w]
            frame = cv2.resize(frame, (width,heigth))
            #mask = cv2.resize(mask, (width,heigth))
            video_croped.append(frame)
            #mask_croped.append(mask)

        elif len(faces_profile) == 1:
            x, y, w, h = faces_profile[0]
            frame = frame[y:y+h, x:x+w]
            #mask = mask[y:y+h, x:x+w]
            frame = cv2.resize(frame, (width,heigth))
            #mask = cv2.resize(mask, (width,heigth))
            video_croped.append(frame)
            #mask_croped.append(mask)
        
        else:
            pass
    
    return video_croped