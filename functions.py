import cv2

def find_face(frame, model):
    #we mask the frame into grey
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #for detecting all teh faces in the frame
    faces = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #now we iterate through all the faces to entour them
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame

def start():
    
    model =cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    #Videocapture, select yopur camera with 0,1,2,3...
    cap = cv2.VideoCapture(0)

    #check the access to camera
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        exit()

    #capture each frame until pressed "q"
    
    while True:
        ret, f = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break
        #call the function to recognize the frame
        find_face(f, model)
        #show the frame with the face marked
        cv2.imshow('Video', f)

        # Check for keypress (press 'q' to exit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #release the video and close the display window
    cap.release()
    cv2.destroyAllWindows()