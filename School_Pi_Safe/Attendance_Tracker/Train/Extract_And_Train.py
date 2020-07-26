#IDEA FOR THIS MODEL WAS BASED OFF OF THIS SOURCE: https://www.pyimagesearch.com/2020/01/06/raspberry-pi-and-movidius-ncs-face-recognition/

import numpy as np
import cv2
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import imutils
from imutils import paths as path
import pickle
import os


#DEFINING GAMMA FUNCTION TO MAKE INPUT IMAGES BRIGHTER:
def gamma(image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

print('Extracting Facial Embeddings')

#CREATE EMPTY LISTS:
Known_Face_Embeddings = []
Known_Names = []

#LOADING EMBEDDING AND FACE DETECTION MODELS:
detector = cv2.dnn.readNetFromCaffe('models/deploy.prototxt','models/res10_300x300_ssd_iter_140000.caffemodel')
embedder = cv2.dnn.readNetFromTorch('models/openface_nn4.small2.v1.t7')


#LOAD IMAGES FROM DATA:
Dataset = 'Data'
imgPaths = path.list_images(Dataset)
imgPaths = list(imgPaths)


#SCAN THROUGH THE DATA
Num_imgPaths = enumerate(imgPaths)

print('Scanning through the Data')
for (i, imagePath) in Num_imgPaths:
        #SEPERATE THE NAME AND APPEND TO LIST
        name = imagePath.split(os.path.sep)[-2]
        Known_Names.append(name)


        #APPLY GAMMA TO IMAGE AND EXTRACT THE DIMENSIONS:
        img = cv2.imread(imagePath)
        img = gamma(img)

        (h, w) = img.shape[:2]


        #Now, construct a blob from the image that we can pass into the model
        imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(img, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)


        detector.setInput(imageBlob)
        detections = detector.forward()


        #Only proceed if at least one face was detected
        if len(detections) > 0:
                #Take face with largest probability
                j = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, j, 2]

        #Continue only if the confidence of the detection exceeds 0.5
                if confidence > 0.5:
                        #Compute the coordinates for the face box
                        box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

            #Grab the face ROI
                        face = img[startY:endY, startX:endX]
                        (H, W) = face.shape[:2]

            #Continue with the loop only if the face height or width is large enough
                        if H < 20 or W < 20:
                                continue

                        #Construct another blob for the face, and send it to the embedder model:
                        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                        embedder.setInput(faceBlob)
                        embeddings = embedder.forward()

                        #Flatten the array
                        embeddings = embeddings.flatten()

                        #Append the Known_Face_Embeddings List with the embedding
                        Known_Face_Embeddings.append(embeddings)



print('Done with embeddings')
#Store the data collected and write it to a pickle
Embeddings_Data = {"embeddings": Known_Face_Embeddings, "names": Known_Names}
f = open('Facial_Embeddings.pickle', "wb")
f.write(pickle.dumps(Embeddings_Data))
f.close()



print('Training the model')

#Reload the pickle for facial embeddings
file = open('Facial_Embeddings.pickle','rb')
data = pickle.load(file)

#Create a Label Encoder object and set the labels to be the names in the Facial_Embeddings.
le = LabelEncoder()
face_labels = le.fit_transform(data["names"])

#Set the embeddings to be the embeddings
face_embeddings = data["embeddings"]


#Set Parameters for the model
params = {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        "gamma": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}

#Build Model Infrastructure
model = GridSearchCV(SVC(kernel="rbf", gamma="auto",
        probability=True), params, cv=3, n_jobs=-1)

#Fit the model
model.fit(face_embeddings, face_labels)


#Store the actual model and the label encoder as pickle files
f = open('Facial_Recognition.pickle', "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

f = open('Facial_Labels.pickle', "wb")
f.write(pickle.dumps(le))
f.close()
