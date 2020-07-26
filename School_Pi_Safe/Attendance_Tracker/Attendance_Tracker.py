import numpy as np
import pickle
import time
import cv2
import pandas as pd
import os
import smtplib,ssl
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders
from datetime import datetime
import RPi.GPIO as gpio


NCS = input('Do you want to use NCS stick?')


MOTOR1_FORWARD_GPIO = 4
ON = 1
OFF = 0

gpio.setmode(gpio.BCM)
gpio.setwarnings(False)
gpio.setup(MOTOR1_FORWARD_GPIO,gpio.OUT)
gpio.output(MOTOR1_FORWARD_GPIO,OFF)


def most_frequent(List): 
	return max(set(List), key = List.count) 

def send_email(excelsheet):
	msg = MIMEMultipart()
	sender_email = "maskdetector101@gmail.com"
	receiver_email = "srinivassriram06@gmail.com"
	password = "LearnIOT06!"
	body = 'Here is an excel sheet which contains the attendance sheet for today'
	msg['From'] = 'maskdetector101@gmail.com'
	msg['To'] = 'srinivassriram06@gmail.com'
	msg['Date'] = formatdate(localtime = True)
	msg['Subject'] = 'Here is the attendance list for today.'

	part = MIMEBase('application', "octet-stream")
	part.set_payload(open(excelsheet, "rb").read())
	encoders.encode_base64(part)
	part.add_header('Content-Disposition', 'attachment; filename="Attendance.xlsx"')
	msg.attach(part)
	msg.attach(MIMEText(body,"plain"))


	context = ssl.create_default_context()
	with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
		server.login(sender_email, password)
		server.sendmail(sender_email, receiver_email, msg.as_string())



df = pd.DataFrame({'Name':['Srinivas','Milan','Abhisar','Aditya'], 'P/A':['Absent','Absent','Absent','Absent']})
names = df['Name']

def check(name):
		counter = -1
		for i in names:
			counter = counter + 1
			if name == i:
				df[r'P/A'][counter] = "Present"
				print("[INFO]: Opening Door...")
				gpio.output(MOTOR1_FORWARD_GPIO, ON)
				time.sleep(3)
				print("[INFO]: Stopping Motor...")
				gpio.output(MOTOR1_FORWARD_GPIO, OFF)
				time.sleep(2)

protoPath = 'models/deploy.prototxt'
modelPath = 'models/res10_300x300_ssd_iter_140000.caffemodel'
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch('models/openface_nn4.small2.v1.t7')
if NCS == 'Yes':
	detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
	embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
recognizerfile = open('models/Facial_Recognition.pickle','rb')
recognizer = pickle.load(recognizerfile)
lefile = open('models/Facial_Labels.pickle','rb')
le = pickle.load(lefile)

now = datetime.now().time()
print("now =", now)
hr = now.hour
min = now.minute
fail_safe = []
stream = cv2.VideoCapture(0)

while True:
	ret, frame = stream.read()

	(h, w) = frame.shape[:2]


	imgBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	detector.setInput(imgBlob)
	detections = detector.forward()


	if len(detections) > 0:
		j = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, j, 2]

		if confidence > 0.5:
			box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			face = frame[startY:endY, startX:endX]
			(H, W) = face.shape[:2]

			if W < 20 or H < 20:
				continue

			faceBlob = cv2.dnn.blobFromImage(cv2.resize(face,
				(96, 96)), 1.0 / 255, (96, 96), (0, 0, 0),
				swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]
			i = 0
			if i < 10:
				fail_safe.append(name)
				i +=1
			most_freq = most_frequent(fail_safe)
			check(most_freq)
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	cv2.imshow("Attendance Tracker", frame)
	key = cv2.waitKey(1) & 0xFF

	now = datetime.now().time()
	hr = now.hour
	min = now.minute

	#CAN CHANGE TIME IF NECESSARY
	if hr == 12 and min == 45:
		print("Exiting due to school start.")
		break
	if key == ord("q"):
		break


Xfilename = "Attendance" + ".xlsx"
df.to_excel(Xfilename)
send_email('Attendance.xlsx')

# do a bit of cleanup
cv2.destroyAllWindows()
stream.release()
