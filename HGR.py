import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import cv2
import imutils


class HGR:

    bg = None
    clone = None
    thresholded = None
    predicted_class = None
    confidence = None

    def __init__(self, video_source=0, model_location="TrainedModel/GestureRecogModel.tfl"):
        tf.reset_default_graph()
        convnet = input_data(shape=[None, 89, 100, 1], name='input')
        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)
        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 128, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 256, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 256, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 128, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = fully_connected(convnet, 1000, activation='relu')
        convnet = dropout(convnet, 0.75)

        convnet = fully_connected(convnet, 7, activation='softmax')

        convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
                             name='regression')

        self.model = tflearn.DNN(convnet, tensorboard_verbose=0)

        # Load Saved Model
        self.model.load(model_location)

        # initialize weight for running average
        self.a_weight = 0.5

        # get the reference to the webcam
        self.camera = cv2.VideoCapture(video_source)

        # region of interest (ROI) coordinates
        self.top, self.right, self.bottom, self.left = 10, 350, 225, 590

        # initialize num of frames
        self.num_frames = 0

    def predict(self):
        predicted_class = None
        confidence = None
        # get the current frame
        (grabbed, frame) = self.camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        self.clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[self.top:self.bottom, self.right:self.left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if self.num_frames < 30:
            self.run_avg(gray, self.a_weight)
        else:
            # segment the hand region
            hand = self.segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (self.thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(self.clone, [segmented + (self.right, self.top)], -1, (0, 0, 255))
                cv2.imwrite('Temp.png', self.thresholded)
                self.resize_image('Temp.png')
                self.predicted_class, self.confidence = self.get_predicted_class()

        # draw the segmented hand
        cv2.rectangle(self.clone, (self.left, self.top), (self.right, self.bottom), (0, 255, 0), 2)

        # increment the number of frames
        self.num_frames += 1

    def get_frame(self):
        return self.clone

    def get_thresholded_hand(self):
        return self.thresholded

    def run_avg(self, image, a_weight):
        # initialize the background
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return

        # compute weighted average, accumulate it and update the background
        cv2.accumulateWeighted(image, self.bg, a_weight)

    def segment(self, image, threshold=25):
        # find the absolute difference between background and current frame
        diff = cv2.absdiff(self.bg.astype("uint8"), image)

        # threshold the diff image so that we get the foreground
        thresholded = cv2.threshold(diff,
                                    threshold,
                                    255,
                                    cv2.THRESH_BINARY)[1]

        # get the contours in the thresholded image
        (cnts, _) = cv2.findContours(thresholded.copy(),
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

        # return None, if no contours detected
        if len(cnts) == 0:
            return
        else:
            # based on contour area, get the maximum contour which is the hand
            segmented = max(cnts, key=cv2.contourArea)
            return thresholded, segmented

    def get_predicted_class(self):
        # Predict
        image = cv2.imread('Temp.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        prediction = self.model.predict([gray_image.reshape(89, 100, 1)])
        predicted_class = np.argmax(prediction)
        if predicted_class == 0:
            class_name = "Palm"
        elif predicted_class == 1:
            class_name = "Fist"
        elif predicted_class == 2:
            class_name = "OK"
        elif predicted_class == 3:
            class_name = "Thumb"
        elif predicted_class == 4:
            class_name = "V"
        elif predicted_class == 5:
            class_name = "L"
        elif predicted_class == 6:
            class_name = "Swing"
        return class_name, (np.amax(prediction) / (
                    prediction[0][0] + prediction[0][1] + prediction[0][2] + prediction[0][3] + prediction[0][4] +
                    prediction[0][5] + prediction[0][6]))

    def resize_image(self, image_name):
        basewidth = 100
        img = Image.open(image_name)
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        img.save(image_name)
