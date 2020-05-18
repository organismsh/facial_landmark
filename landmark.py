from imutils import face_utils
import dlib
import cv2
import numpy as np

#pretrained model for detecting facial landmarks
pre_trained_model = "classifier/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pre_trained_model)

#loading video input for facial landmark extraction
video = cv2.VideoCapture('video/video.mp4')

while video.read():
    #getting video from the file
    _, image_input = video.read()

    #resizing video
    resize = cv2.resize(image_input, (1050,600)) 
    image = resize
    
    # Converting the image to gray scale
    out_face = np.zeros_like(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Get faces into webcam's image
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #plot landmark points in detected face 
        for (x, y) in shape:
            cv2.circle(image, (x, y), 4, (0, 255, 5), -5)
    
        #face extraction
        remapped_shape = np.zeros_like(shape) 
        feature_mask = np.zeros((image.shape[0], image.shape[1]))   
        remapped_shape = cv2.convexHull(shape)
        cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
        feature_mask = feature_mask.astype(np.bool)
        out_face[feature_mask] = image[feature_mask]


    #output window
    cv2.imshow("Output", out_face)
    cv2.resizeWindow('Output', 50,50)    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()