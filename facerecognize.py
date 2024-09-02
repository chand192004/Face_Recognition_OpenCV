#1.Generate Dataset
#2.Train the classifier and save it
#3.Detect the face and name it if it is already stored in our dataset

import tkinter as tk
from tkinter import messagebox
import cv2
import os
from PIL import Image
import numpy as np

window=tk.Tk()
# window.config(background="lime")
window.title("Face recognition system")

l1=tk.Label(window,text="Name",font=("Algerian",20))
l1.grid(column=0, row=0)
t1=tk.Entry(window,width=50,bd=5)
t1.grid(column=1, row=0)

l2=tk.Label(window,text="Age",font=("Algerian",20))
l2.grid(column=0, row=1)
t2=tk.Entry(window,width=50,bd=5)
t2.grid(column=1, row=1)

l3=tk.Label(window,text="User ID",font=("Algerian",20))
l3.grid(column=0, row=2)
t3=tk.Entry(window,width=50,bd=5)
t3.grid(column=1, row=2)


def train_classifier():
    collectData_dir="C:/Users/DELL/Documents/FaceRecognitionProject/collectData"
    path = [os.path.join(collectData_dir,f) for f in os.listdir(collectData_dir)]
    faces = []
    ids = []
    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])    
        
        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)
    #Train the classifier and save
    clf= cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.xml")
    messagebox.showinfo('Result','Training Dataset Completed!')    
        
        
b1=tk.Button(window,text="Training",font=("Algerian",20),bg='orange',fg='red',command=train_classifier)
b1.grid(column=0, row=4)

def detect_face():
    def Draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
        coords = []
        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))
            if confidence > 70:
                if id == 1:
                    cv2.putText(img, "Chandni", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                if id == 2:
                    cv2.putText(img, "Shakshi", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                if id == 3:
                    cv2.putText(img, "Virat Kohli", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                if id == 4:
                    cv2.putText(img, "Kalpana Chawla", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                    cv2.putText(img, "Unknown", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            coords.append((x, y, w, h))
        return coords

    def recognize(img, clf, faceCascade):
        coords = Draw_boundary(img, faceCascade, 1.1, 10, (50, 205, 50), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0)
    while True:
        ret, img = video_capture.read()
        img = recognize(img, clf, faceCascade)
        cv2.imshow("Face Detection", img)

        if cv2.waitKey(1) == 13:  # Enter key to break
            break

    video_capture.release()
    cv2.destroyAllWindows()
       
       
b2=tk.Button(window,text="Detect the face",font=("Algerian",20),bg='green',fg='white',command=detect_face)
b2.grid(column=1, row=4)

def generate_dataset():
    if(t1.get()=="" or t2.get()=="" or t3.get()==""):
        messagebox.showinfo('Result','Please provide complete details of the user')
    else:
        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        def face_crop(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray,1.3, 5)
            # scaling factor = 1.3
            # Minimum neighbor = 5

            if len(faces) == 0:
                return None
            for (x, y, w, h) in faces:
                cropped_face = img[y:y+h, x:x+w]
                return cropped_face

        cap = cv2.VideoCapture(0)
        id = 1
        img_id = 0

        while True:
            ret, frame = cap.read()
            if ret:
                face = face_crop(frame)
                if face is not None:
                    img_id += 1
                    face = cv2.resize(face, (200, 200))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    file_name_path = "collectData/user." + str(id) + "." + str(img_id) + ".jpg"
                    cv2.imwrite(file_name_path, face)
                    cv2.putText(face,str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    # (50, 50) is the origin point from where text is to be written
                    # font scale = 1
                    # thickness = 2 

                    cv2.imshow("Cropped Face", face)
                    if cv2.waitKey(1) == 13 or int(img_id) == 200:
                        break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result','Generating Dataset completed!')
b3=tk.Button(window,text="Generate Dataset",font=("Algerian",20),bg='pink',fg='black',command=generate_dataset)
b3.grid(column=2, row=4)


window.geometry("800x200")
window.mainloop()



