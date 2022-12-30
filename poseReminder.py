import cv2
import PIL.Image, PIL.ImageTk
import datetime
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
import dlib
import tkinter as tk
from tkinter import ttk, Tk, Canvas, Entry, Text, Button, PhotoImage, Frame, NW, messagebox
from pathlib import Path
import os
import sys



OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")

wink_reminder = """ osascript -e '
  display notification "您的眨眼次數太少囉，請記得多眨眼" with title "PoseReminder" sound name "Crystal"
'"""
left_reminder = """ osascript -e '
  display notification "您的頭部有點往左偏囉，請回正" with title "PoseReminder" sound name "Crystal"
'"""
right_reminder = """ osascript -e '
  display notification "您的頭部有點往右偏囉，請回正" with title "PoseReminder" sound name "Crystal"
'"""
up_reminder = """ osascript -e '
  display notification "您的頭部有點往上抬囉，請回正" with title "PoseReminder" sound name "Crystal"
'"""
down_reminder = """ osascript -e '
  display notification "您的頭部有點往下降囉，請回正" with title "PoseReminder" sound name "Crystal"
'"""
front_reminder = """ osascript -e '
  display notification "您的頭部太靠近電腦囉，請回正" with title "PoseReminder" sound name "Crystal"
'"""

def my_path(path_name):
    """Return the appropriate path for data files based on execution context"""
    if getattr( sys, 'frozen', False ):
        # running in a bundle
        return(os.path.join(sys._MEIPASS, path_name))
    else:
        # running live
        return path_name

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.18
EYE_AR_CONSEC_FRAMES = 3

FRAME_TIME = 25
THRESHOLD = 3
EYE_FRAME_TIME = 50 # 15s
EYE_THRESHOLD = 4


#儲存標準位置(68點+左上與右下座標)
standard = []
standard_flag = 0
shapeF = []

for i in range(70):
  row = []
  for j in range(2):
    row.append(0)
  standard.append(row)
  shapeF.append(row)

flag = []
for i in range(6):
  row = []
  flag.append(row)

flagAll = []

model_name = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(my_path(model_name))
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


class MyVideoCapture:
  def __init__(self, video_source=0):
    # Open the video source
    self.vid = cv2.VideoCapture(video_source)
    if not self.vid.isOpened():
      raise ValueError("Unable to open video source", video_source)

       # Get video source width and height
    self.width = 1280
    self.height = 720

  def get_frame(self):
    if self.vid.isOpened():
      # read frames
      ret, frame = self.vid.read()
      # resize
      frame = imutils.resize(frame, width=self.width,height=self.height)

      #偵測人臉
      face_rects, scores, idx = detector.run(frame, 0)

      global standard_flag
      level = 0
      #取出偵測的結果
      for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        # text = "%2.2f ( %d )" % (scores[i], idx[i])
        #繪製出偵測人臉的矩形範圍
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

        #標上人臉偵測分數與人臉子偵測器編號
        # cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
 
        #給68特徵點辨識取得一個轉換顏色的frame
        landmarks_frame = cv2.cvtColor(frame, cv2. COLOR_BGR2RGB)

        #找出特徵點位置
        shape = predictor(landmarks_frame, d)

        #繪製68個特徵點
        for i in range(68):
            cv2.circle(frame,(shape.part(i).x,shape.part(i).y), 1,( 0, 0, 255), 1)
            # cv2.putText(frame, str(i),(shape.part(i).x,shape.part(i).y),cv2.FONT_HERSHEY_COMPLEX, 0.5,( 255, 0, 0), 1)
            shapeF[i][0]=shape.part(i).x
            shapeF[i][1]=shape.part(i).y

        shapeF[68] = [x1,y1]
        shapeF[69] = [x2,y2]
        #shape.part(i).x, shape.part(i).y

        #判斷是否已儲存標準位置
        if (standard_flag):
        #依鼻子和雙眼關係變化程度
          #頭部左移
          if((shape.part(42).x-shape.part(27).x) / float(standard[42][0]-standard[27][0]) <0.8):
            flag[0].append(1)
          else:
            flag[0].append(0)
          #頭部右移
          if((shape.part(39).x-shape.part(27).x) / float(standard[39][0]-standard[27][0]) <0.8):
              flag[1].append(1)
          else:
              flag[1].append(0)
        #依下巴與鼻子長度關係
          #頭部上抬
          if(shape.part(39).y - standard[39][1] < -15):
            flag[2].append(1)
          else:
            flag[2].append(0)
          #頭部下降
          if(shape.part(39).y - standard[39][1] > 20):
            flag[3].append(1)
          else:
            flag[3].append(0)
        #依照面部大小變化
          #頭部前傾
          if(pow(x2-x1, 2) / float(pow((standard[69][0]-standard[68][0]), 2)) > 1.07):
            flag[4].append(1)
          else:
            flag[4].append(0)
        
          
    #   # turn to model img type
    #   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #   rects = detector(gray, 0)
    #   for rect in rects:
    #     shape = predictor(gray, rect)
        shape_e = face_utils.shape_to_np(shape)
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape_e[lStart:lEnd]
        rightEye = shape_e[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
		# average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

    # compute the convex hull for the left and right eye, then
		# visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        # check to see if the eye aspect ratio is below the blink
		    # threshold, and if so, increment the blink frame counter
        # initialize the frame counters and the total number of blinks
        # COUNTER = 0
    #     if ear < EYE_AR_THRESH:
    #       COUNTER += 1
		# # otherwise, the eye aspect ratio is not below the blink
		# # threshold
    #     else:
		# 	# if the eyes were closed for a sufficient number of
		# 	# then increment the total number of blinks
    #       if COUNTER >= EYE_AR_CONSEC_FRAMES:
    #         flag[5].append(1)
    #       else:
    #         flag[5].append(0)
			# reset the eye frame counter
          # COUNTER = 0
        if ear < EYE_AR_THRESH:
          flag[5].append(1)
        else:
          flag[5].append(0)

        # print(datetime.datetime.now(),len(flag[5]),sum(flag[5]))
        if(len(flag[5])>EYE_FRAME_TIME):
          flag[5].pop(0)
          if(sum(flag[5]) < EYE_THRESHOLD):
            flag[5] = []
            os.system(wink_reminder)

        global FRAME_TIME
        global THRESHOLD
        # print(datetime.datetime.now(),len(flag[0]),sum(flag[0]))
      #依照flag紀錄print出目前狀況
        if(len(flag[0])>FRAME_TIME):
          if(sum(flag[0]) > THRESHOLD):
            flag[0] = []
            os.system(left_reminder)
          else:
            flag[0].pop(0)

          if(sum(flag[1]) > THRESHOLD):
            flag[1] = []
            os.system(right_reminder)
          else:
            flag[1].pop(0)

          if(sum(flag[2]) > THRESHOLD):
            flag[2] = []
            os.system(up_reminder)
          else:
            flag[2].pop(0)

          if(sum(flag[3]) > THRESHOLD):
            flag[3] = []
            os.system(down_reminder)
          else:
            flag[3].pop(0)

          if(sum(flag[4]) > THRESHOLD):
            flag[4] = []
            os.system(front_reminder)
          else:
            flag[4].pop(0)
       

      # draw the total number of blinks on the frame along with
		  # the computed eye aspect ratio for the frame
      # cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			# cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
      # cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			# cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


      if ret:
        # Return a boolean success flag and the current frame converted to BGR
        return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      else:
        return (ret, None)
    else:
      return -1
 
     # Release the video source when the object is destroyed
  def storeData(self):
    global standard_flag
    standard_flag = 1
    #標準位置，儲存資料
    for i in range(70):
      standard[i] = ([shapeF[i][0], shapeF[i][1]])
    

  def __del__(self):
    if self.vid.isOpened():
      self.vid.release()

soundflag = 1

class App:
  def __init__(self, window, window_title):
    self.window = window

    self.window.geometry("1280x720")
    self.window.title(window_title)
    self.tips()

  def back(self):
    self.window.after_cancel(self.refresh)
    self.vid.__del__()
    self.welcome()

  def tips(self):
    for i in self.window.winfo_children():
      i.destroy()
    self.canvas = Canvas(
            self.window,
            bg = "#FFFFFF",
            height = 720,
            width = 1280,
            bd = 0,
            highlightthickness = 0,
            relief = "flat"
    )
    button_tips = Button(
            image=button_image_tips,
            borderwidth=0,
            highlightthickness=0,
            command=self.welcome,
            relief="flat"
        )
    button_tips.place(
            x=0,
            y=0,
            width=1280,
            height=720
        )
    self.canvas.place(x = 0, y = 0)

  def welcome(self):
        for i in self.window.winfo_children():
            i.destroy()

        global standard_flag
        standard_flag = 0
        
        self.canvas = Canvas(
            self.window,
            bg = "#FFFFFF",
            height = 720,
            width = 1280,
            bd = 0,
            highlightthickness = 0,
            relief = "flat"
        )

        
        image_welcome = self.canvas.create_image(
            574.0,
            388.99999999999955,
            image=image_image_welcome
        )
        

        self.canvas.create_text(
            355.0,
            622.9999999999995,
            anchor="nw",
            text="Please click the icon to use Pose Reminder",
            fill="#FFFFFF",
            font=("Inter ExtraLight", 32 * -1)
        )

        self.canvas.create_text(
            270.0,
            127.99999999999955,
            anchor="nw",
            text="Welcome to Pose Reminder",
            fill="#FFFFFF",
            font=("Inter Bold", 64 * -1)
        )

        
        button_noMask = Button(
            image=button_image_noMask,
            borderwidth=0,
            highlightthickness=0,
            command=self.no_mask,
            relief="flat",
            cursor="man"
        )
        button_noMask.place(
            x=291.0,
            y=246.99999999999955,
            width=275.0,
            height=340.0
        )

        
        button_mask = Button(
            image=button_image_mask,
            borderwidth=0,
            highlightthickness=0,
            command=self.mask,
            relief="flat",
            cursor="man"
        )
        button_mask.place(
            x=716.0,
            y=246.99999999999955,
            width=275.0,
            height=340.0
        )
        self.canvas.place(x = 0, y = 0)

  def setmode1(self):
    global THRESHOLD
    global FRAME_TIME
    THRESHOLD = 3
    FRAME_TIME = 25

  def setmode2(self):
    global THRESHOLD
    global FRAME_TIME
    THRESHOLD = 50
    FRAME_TIME = 250

  def setmode3(self):
    global THRESHOLD
    global FRAME_TIME
    THRESHOLD = 150
    FRAME_TIME = 750

  def sound(self):
    global wink_reminder,left_reminder,right_reminder,up_reminder,down_reminder,front_reminder
    global soundflag
    if soundflag == 0:
      self.button_sound.configure(image=button_image_sound)
      soundflag = 1
      wink_reminder = """ osascript -e '
        display notification "您的眨眼次數太少囉，請記得多眨眼" with title "PoseReminder" sound name "Crystal"
      '"""
      left_reminder = """ osascript -e '
        display notification "您的頭部有點往左偏囉，請回正" with title "PoseReminder" sound name "Crystal"
      '"""
      right_reminder = """ osascript -e '
        display notification "您的頭部有點往右偏囉，請回正" with title "PoseReminder" sound name "Crystal"
      '"""
      up_reminder = """ osascript -e '
        display notification "您的頭部有點往上抬囉，請回正" with title "PoseReminder" sound name "Crystal"
      '"""
      down_reminder = """ osascript -e '
        display notification "您的頭部有點往下降囉，請回正" with title "PoseReminder" sound name "Crystal"
      '"""
      front_reminder = """ osascript -e '
        display notification "您的頭部太靠近電腦囉，請回正" with title "PoseReminder" sound name "Crystal"
      '"""
      
    else:
      self.button_sound.configure(image=button_image_nosound)
      soundflag = 0
      wink_reminder = """ osascript -e '
        display notification "您的眨眼次數太少囉，請記得多眨眼" with title "PoseReminder"
      '"""
      left_reminder = """ osascript -e '
        display notification "您的頭部有點往左偏囉，請回正" with title "PoseReminder"
      '"""
      right_reminder = """ osascript -e '
        display notification "您的頭部有點往右偏囉，請回正" with title "PoseReminder"
      '"""
      up_reminder = """ osascript -e '
        display notification "您的頭部有點往上抬囉，請回正" with title "PoseReminder"
      '"""
      down_reminder = """ osascript -e '
        display notification "您的頭部有點往下降囉，請回正" with title "PoseReminder"
      '"""
      front_reminder = """ osascript -e '
        display notification "您的頭部太靠近電腦囉，請回正" with title "PoseReminder"
      '"""
  
  def start(self):
    self.button_start.configure(image=button_image_start1)
    self.vid.storeData()
    

  def no_mask(self):
    for i in self.window.winfo_children():
      i.destroy()

    self.video_source = 0

    # open video source (by default this will try to open the computer webcam)
    self.vid = MyVideoCapture(self.video_source)

    # Create a canvas that can fit the above video source size
    self.canvas = tk.Canvas(self.window, width = 1280, height = 720)
    self.canvas.pack()

    global THRESHOLD
    moderadiobutton_1 = tk.Radiobutton(self.window, text="嚴格模式", variable=THRESHOLD,value=5,command=self.setmode1)
    moderadiobutton_2 = tk.Radiobutton(self.window, text="普通模式", variable=THRESHOLD,value=50,command=self.setmode2)
    moderadiobutton_3 = tk.Radiobutton(self.window, text="放鬆模式", variable=THRESHOLD,value=1000,command=self.setmode3)
    moderadiobutton_1.pack()
    moderadiobutton_1.place(x=10,y=10)
    moderadiobutton_2.pack()
    moderadiobutton_2.place(x=10,y=30)
    moderadiobutton_3.pack()
    moderadiobutton_3.place(x=10,y=50)
    moderadiobutton_1.select()
    
    # Button that lets the user can back to mode choose
    button_back = Button(
      image=button_image_back,
      borderwidth=0,
      highlightthickness=0,
      command=self.back,
      relief="raised"
    )
    button_back.place(
      x=10.0,
      y=100.0,
      width=89.0,
      height=89.0
    )

    self.button_start = Button(
      image=button_image_start,
      borderwidth=0,
      highlightthickness=0,
      command=self.start,
      relief="raised"
    )
    self.button_start.place(
      x=10.0,
      y=313.0,
      width=89.0,
      height=89.0
    )

    self.button_sound = Button(
      image=button_image_sound,
      borderwidth=0,
      highlightthickness=0,
      command=self.sound,
      relief="raised"
    )
    self.button_sound.place(
      x=10.0,
      y=526.0,
      width=89.0,
      height=89.0
    )

    # After it is called once, the update method will be automatically called every delay milliseconds
    self.delay = 10
    self.update()

  def mask(self):
    for i in self.window.winfo_children():
      i.destroy()

    self.video_source = 0

    # open video source (by default this will try to open the computer webcam)
    self.vid = MyVideoCapture(self.video_source)

    # Create a canvas that can fit the above video source size
    self.canvas = tk.Canvas(self.window, width = 1280, height = 720)
    self.canvas.pack()

    global THRESHOLD
    moderadiobutton_1 = tk.Radiobutton(self.window, text="嚴格模式", variable=THRESHOLD,value=5,command=self.setmode1)
    moderadiobutton_2 = tk.Radiobutton(self.window, text="普通模式", variable=THRESHOLD,value=50,command=self.setmode2)
    moderadiobutton_3 = tk.Radiobutton(self.window, text="放鬆模式", variable=THRESHOLD,value=1000,command=self.setmode3)
    moderadiobutton_1.pack()
    moderadiobutton_1.place(x=10,y=10)
    moderadiobutton_2.pack()
    moderadiobutton_2.place(x=10,y=30)
    moderadiobutton_3.pack()
    moderadiobutton_3.place(x=10,y=50)
    moderadiobutton_1.select()
    
    # Button that lets the user can back to mode choose
    button_back = Button(
      image=button_image_back,
      borderwidth=0,
      highlightthickness=0,
      command=self.back,
      relief="raised"
    )
    button_back.place(
      x=10.0,
      y=100.0,
      width=89.0,
      height=89.0
    )

    self.button_start = Button(
      image=button_image_start,
      borderwidth=0,
      highlightthickness=0,
      command=self.start,
      relief="raised"
    )
    self.button_start.place(
      x=10.0,
      y=313.0,
      width=89.0,
      height=89.0
    )

    self.button_sound = Button(
      image=button_image_sound,
      borderwidth=0,
      highlightthickness=0,
      command=self.sound,
      relief="raised"
    )
    self.button_sound.place(
      x=10.0,
      y=526.0,
      width=89.0,
      height=89.0
    )

    # After it is called once, the update method will be automatically called every delay milliseconds
    self.delay = 1
    self.update()
    

  def update(self):
    # Get a frame from the video source
    ret, frame = self.vid.get_frame()
 
    if ret:
      self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
      self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)

    self.refresh = self.window.after(self.delay, self.update)


root = tk.Tk()

button_image_tips = PhotoImage(file=my_path("button_tip.png"))

image_image_welcome = PhotoImage(file=my_path("image_welcome.png"))
button_image_noMask = PhotoImage(file=my_path("button_noMask.png"))
button_image_mask = PhotoImage(file=my_path("button_mask.png"))

img_back = PIL.Image.open(my_path("back.png"))
img_back = img_back.resize((80,80), PIL.Image.ANTIALIAS)
button_image_back =  PIL.ImageTk.PhotoImage(img_back)

img_start = PIL.Image.open(my_path("start.png"))
img_start = img_start.resize((80,80), PIL.Image.ANTIALIAS)
button_image_start =  PIL.ImageTk.PhotoImage(img_start)

img_start1 = PIL.Image.open(my_path("start_1.png"))
img_start1 = img_start1.resize((80,80), PIL.Image.ANTIALIAS)
button_image_start1 =  PIL.ImageTk.PhotoImage(img_start1)

img_sound = PIL.Image.open(my_path("sound.png"))
img_sound = img_sound.resize((80,80), PIL.Image.ANTIALIAS)
button_image_sound =  PIL.ImageTk.PhotoImage(img_sound)

img_nosound = PIL.Image.open(my_path("no_sound.png"))
img_nosound = img_nosound.resize((80,80), PIL.Image.ANTIALIAS)
button_image_nosound =  PIL.ImageTk.PhotoImage(img_nosound)


# Create a window and pass it to the Application object
App(root, "Pose Reminder")

root.resizable(width=False,height=False)

root.mainloop()
