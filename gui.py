from tkinter import Tk, Label, Button, StringVar
# USAGE
# python social_distance_detector.py --input pedestrians.mp4
# python social_distance_detector.py --input pedestrians.mp4 --output output.avi

# import the necessary packages
from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
from mss import mss



class MSSSource:
    def __init__(self):
        self.sct = mss()

    def frame(self, width, height):
      monitor = {'top': 0, 'left': 0, 'width': width, 'height': height}
      im = np.array(self.sct.grab(monitor))
      im = np.flip(im[:, :, :3], 2)  # 1
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 2
      return True, im

    def release(self):
        pass




class Vision2020:
    
    def __init__(self, master):
        self.master = master
        master.title("2020vision program")
        
        
        self.label_index = 0
        self.label_text = StringVar()
        self.label_text.set("Welcome to the 2020vision software.")
        self.label = Label(master, textvariable=self.label_text)
        self.label.bind("<Button-1>", self.cycle_label_text)
        self.label.pack()

        

    def fetch(entries):
        for entry in entries:
            field = entry[0]
            text  = entry[1].get()
            print('%s: "%s"' % (field, text)) 

    def makeform(root, fields):
        entries = []
        for field in fields:
            row = tk.Frame(root)
            lab = tk.Label(row, width=15, text=field, anchor='w')
            ent = tk.Entry(row)
            row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
            lab.pack(side=tk.LEFT)
            ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
            entries.append((field, ent))
        return entries

    if __name__ == '__main__':
        root = tk.Tk()
        ents = makeform(root, fields)
        root.bind('<Return>', (lambda event, e=ents: fetch(e)))   
        b1 = tk.Button(root, text='Show',
                      command=(lambda e=ents: fetch(e)))
        b1.pack(side=tk.LEFT, padx=5, pady=5)
        b2 = tk.Button(root, text='Quit', command=root.quit)
        b2.pack(side=tk.LEFT, padx=5, pady=5)
        root.mainloop()


        self.greet_button = Button(master, text="Start", command=self.start)
        self.greet_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    def greet(self):
        print("Greetings!")

    def start(self):
        source = MSSSource()

        # construct the argument parse and parse the arguments
        # ap = argparse.ArgumentParser()
        # ap.add_argument("-i", "--input", type=str, default="",
        #   help="path to (optional) input video file")
        # ap.add_argument("-o", "--output", type=str, default="",
        #   help="path to (optional) output video file")
        # ap.add_argument("-d", "--display", type=int, default=1,
        #   help="whether or not output frame should be displayed")
        # args = vars(ap.parse_args())

        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
        LABELS = open(labelsPath).read().strip().split("\n")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
        configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        # check if we are going to use GPU
        if config.USE_GPU:
          # set CUDA as the preferable backend and target
          print("[INFO] setting preferable backend and target to CUDA...")
          net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
          net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # initialize the video stream and pointer to output video file
        ###print("[INFO] accessing video stream...")
        #v#s = cv2.VideoCapture("pmbstest.mp4")
        # vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
        writer = None

        # loop over the frames from the video stream
        while True:
          # read the next frame from the file
          (grabbed, frame) = source.frame()

          # if the frame was not grabbed, then we have reached the end
          # of the stream
          if not grabbed:
            break

          # resize the frame and then detect people (and only people) in it
          frame = imutils.resize(frame, width=700)
          results = detect_people(frame, net, ln,
            personIdx=LABELS.index("person"))

          # initialize the set of indexes that violate the minimum social
          # distance
          violate = set()

          # ensure there are *at least* two people detections (required in
          # order to compute our pairwise distance maps)
          if len(results) >= 2:
            # extract all centroids from the results and compute the
            # Euclidean distances between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
              for j in range(i + 1, D.shape[1]):
                # check to see if the distance between any two
                # centroid pairs is less than the configured number
                # of pixels
                if D[i, j] < config.MIN_DISTANCE:
                  # update our violation set with the indexes of
                  # the centroid pairs
                  violate.add(i)
                  violate.add(j)

          # loop over the results
          for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract the bounding box and centroid coordinates, then
            # initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            # if the index pair exists within the violation set, then
            # update the color
            if i in violate:
              color = (0, 0, 255)

            # draw (1) a bounding box around the person and (2) the
            # centroid coordinates of the person,
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

          # draw the total number of social distancing violations on the
          # output frame
          text = "Social Distancing Violations: {}".format(len(violate))
          cv2.putText(frame, text, (10, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

          # check to see if the output frame should be displayed to our
          # screen
          if args["display"] > 0:
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
              break

          # if an output video file path has been supplied and the video
          # writer has not been initialized, do so now
          if args["output"] != "" and writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 25,
              (frame.shape[1], frame.shape[0]), True)

          # if the video writer is not None, write the frame to the output
          # video file
          if writer is not None:
            writer.write(frame)

 
        ap.add_argument("-d", "--display", type=int, default=1,
          help="whether or not output frame should be displayed")
        args = vars(ap.parse_args())

        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
        LABELS = open(labelsPath).read().strip().split("\n")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
        configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        # check if we are going to use GPU
        if config.USE_GPU:
          # set CUDA as the preferable backend and target
          print("[INFO] setting preferable backend and target to CUDA...")
          net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
          net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # initialize the video stream and pointer to output video file
        ###print("[INFO] accessing video stream...")
        #v#s = cv2.VideoCapture("pmbstest.mp4")
        # vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
        writer = None

        # loop over the frames from the video stream
        while True:
          # read the next frame from the file
          (grabbed, frame) = source.frame()

          # if the frame was not grabbed, then we have reached the end
          # of the stream
          if not grabbed:
            break

          # resize the frame and then detect people (and only people) in it
          frame = imutils.resize(frame, width=700)
          results = detect_people(frame, net, ln,
            personIdx=LABELS.index("person"))

          # initialize the set of indexes that violate the minimum social
          # distance
          violate = set()

          # ensure there are *at least* two people detections (required in
          # order to compute our pairwise distance maps)
          if len(results) >= 2:
            # extract all centroids from the results and compute the
            # Euclidean distances between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
              for j in range(i + 1, D.shape[1]):
                # check to see if the distance between any two
                # centroid pairs is less than the configured number
                # of pixels
                if D[i, j] < config.MIN_DISTANCE:
                  # update our violation set with the indexes of
                  # the centroid pairs
                  violate.add(i)
                  violate.add(j)

          # loop over the results
          for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract the bounding box and centroid coordinates, then
            # initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            # if the index pair exists within the violation set, then
            # update the color
            if i in violate:
              color = (0, 0, 255)

            # draw (1) a bounding box around the person and (2) the
            # centroid coordinates of the person,
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

          # draw the total number of social distancing violations on the
          # output frame
          text = "Social Distancing Violations: {}".format(len(violate))
          cv2.putText(frame, text, (10, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

          # check to see if the output frame should be displayed to our
          # screen
          if args["display"] > 0:
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
              break

          # if an output video file path has been supplied and the video
          # writer has not been initialized, do so now
          if args["output"] != "" and writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 25,
              (frame.shape[1], frame.shape[0]), True)

          # if the video writer is not None, write the frame to the output
          # video file
          if writer is not None:
            writer.write(frame)

 
        ap.add_argument("-d", "--display", type=int, default=1,
          help="whether or not output frame should be displayed")
        args = vars(ap.parse_args())

        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
        LABELS = open(labelsPath).read().strip().split("\n")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
        configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        # check if we are going to use GPU
        if config.USE_GPU:
          # set CUDA as the preferable backend and target
          print("[INFO] setting preferable backend and target to CUDA...")
          net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
          net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # initialize the video stream and pointer to output video file
        ###print("[INFO] accessing video stream...")
        #v#s = cv2.VideoCapture("pmbstest.mp4")
        # vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
        writer = None

        # loop over the frames from the video stream
        while True:
          # read the next frame from the file
          (grabbed, frame) = source.frame()

          # if the frame was not grabbed, then we have reached the end
          # of the stream
          if not grabbed:
            break

          # resize the frame and then detect people (and only people) in it
          frame = imutils.resize(frame, width=700)
          results = detect_people(frame, net, ln,
            personIdx=LABELS.index("person"))

          # initialize the set of indexes that violate the minimum social
          # distance
          violate = set()

          # ensure there are *at least* two people detections (required in
          # order to compute our pairwise distance maps)
          if len(results) >= 2:
            # extract all centroids from the results and compute the
            # Euclidean distances between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
              for j in range(i + 1, D.shape[1]):
                # check to see if the distance between any two
                # centroid pairs is less than the configured number
                # of pixels
                if D[i, j] < config.MIN_DISTANCE:
                  # update our violation set with the indexes of
                  # the centroid pairs
                  violate.add(i)
                  violate.add(j)

          # loop over the results
          for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract the bounding box and centroid coordinates, then
            # initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            # if the index pair exists within the violation set, then
            # update the color
            if i in violate:
              color = (0, 0, 255)

            # draw (1) a bounding box around the person and (2) the
            # centroid coordinates of the person,
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

          # draw the total number of social distancing violations on the
          # output frame
          text = "Social Distancing Violations: {}".format(len(violate))
          cv2.putText(frame, text, (10, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

          # check to see if the output frame should be displayed to our
          # screen
          if args["display"] > 0:
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
              break

          # if an output video file path has been supplied and the video
          # writer has not been initialized, do so now
          if args["output"] != "" and writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 25,
              (frame.shape[1], frame.shape[0]), True)

          # if the video writer is not None, write the frame to the output
          # video file
          if writer is not None:
            writer.write(frame)

 
        ap.add_argument("-d", "--display", type=int, default=1,
          help="whether or not output frame should be displayed")
        args = vars(ap.parse_args())

        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
        LABELS = open(labelsPath).read().strip().split("\n")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
        configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        # check if we are going to use GPU
        if config.USE_GPU:
          # set CUDA as the preferable backend and target
          print("[INFO] setting preferable backend and target to CUDA...")
          net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
          net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # initialize the video stream and pointer to output video file
        ###print("[INFO] accessing video stream...")
        #v#s = cv2.VideoCapture("pmbstest.mp4")
        # vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
        writer = None

        # loop over the frames from the video stream
        while True:
          # read the next frame from the file
          (grabbed, frame) = source.frame()

          # if the frame was not grabbed, then we have reached the end
          # of the stream
          if not grabbed:
            break

          # resize the frame and then detect people (and only people) in it
          frame = imutils.resize(frame, width=700)
          results = detect_people(frame, net, ln,
            personIdx=LABELS.index("person"))

          # initialize the set of indexes that violate the minimum social
          # distance
          violate = set()

          # ensure there are *at least* two people detections (required in
          # order to compute our pairwise distance maps)
          if len(results) >= 2:
            # extract all centroids from the results and compute the
            # Euclidean distances between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
              for j in range(i + 1, D.shape[1]):
                # check to see if the distance between any two
                # centroid pairs is less than the configured number
                # of pixels
                if D[i, j] < config.MIN_DISTANCE:
                  # update our violation set with the indexes of
                  # the centroid pairs
                  violate.add(i)
                  violate.add(j)

          # loop over the results
          for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract the bounding box and centroid coordinates, then
            # initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            # if the index pair exists within the violation set, then
            # update the color
            if i in violate:
              color = (0, 0, 255)

            # draw (1) a bounding box around the person and (2) the
            # centroid coordinates of the person,
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

          # draw the total number of social distancing violations on the
          # output frame
          text = "Social Distancing Violations: {}".format(len(violate))
          cv2.putText(frame, text, (10, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

          # check to see if the output frame should be displayed to our
          # screen
          if args["display"] > 0:
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
              break

          # if an output video file path has been supplied and the video
          # writer has not been initialized, do so now
          if args["output"] != "" and writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 25,
              (frame.shape[1], frame.shape[0]), True)

          # if the video writer is not None, write the frame to the output
          # video file
          if writer is not None:
            writer.write(frame)

 
        ap.add_argument("-d", "--display", type=int, default=1,
          help="whether or not output frame should be displayed")
        args = vars(ap.parse_args())

        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
        LABELS = open(labelsPath).read().strip().split("\n")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
        configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        # check if we are going to use GPU
        if config.USE_GPU:
          # set CUDA as the preferable backend and target
          print("[INFO] setting preferable backend and target to CUDA...")
          net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
          net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # initialize the video stream and pointer to output video file
        ###print("[INFO] accessing video stream...")
        #v#s = cv2.VideoCapture("pmbstest.mp4")
        # vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
        writer = None

        # loop over the frames from the video stream
        while True:
          # read the next frame from the file
          (grabbed, frame) = source.frame()

          # if the frame was not grabbed, then we have reached the end
          # of the stream
          if not grabbed:
            break

          # resize the frame and then detect people (and only people) in it
          frame = imutils.resize(frame, width=700)
          results = detect_people(frame, net, ln,
            personIdx=LABELS.index("person"))

          # initialize the set of indexes that violate the minimum social
          # distance
          violate = set()

          # ensure there are *at least* two people detections (required in
          # order to compute our pairwise distance maps)
          if len(results) >= 2:
            # extract all centroids from the results and compute the
            # Euclidean distances between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
              for j in range(i + 1, D.shape[1]):
                # check to see if the distance between any two
                # centroid pairs is less than the configured number
                # of pixels
                if D[i, j] < config.MIN_DISTANCE:
                  # update our violation set with the indexes of
                  # the centroid pairs
                  violate.add(i)
                  violate.add(j)

          # loop over the results
          for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract the bounding box and centroid coordinates, then
            # initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            # if the index pair exists within the violation set, then
            # update the color
            if i in violate:
              color = (0, 0, 255)

            # draw (1) a bounding box around the person and (2) the
            # centroid coordinates of the person,
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

          # draw the total number of social distancing violations on the
          # output frame
          text = "Social Distancing Violations: {}".format(len(violate))
          cv2.putText(frame, text, (10, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

          # check to see if the output frame should be displayed to our
          # screen
          if args["display"] > 0:
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
              break

          # if an output video file path has been supplied and the video
          # writer has not been initialized, do so now
          if args["output"] != "" and writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 25,
              (frame.shape[1], frame.shape[0]), True)

          # if the video writer is not None, write the frame to the output
          # video file
          if writer is not None:
            writer.write(frame)

 
    def cycle_label_text(self, event):
        self.label_index += 1
        self.label_index %= len(self.LABEL_TEXT) # wrap around
        self.label_text.set(self.LABEL_TEXT[self.label_index])

root = Tk()
my_gui = Vision2020(root)
root.mainloop()