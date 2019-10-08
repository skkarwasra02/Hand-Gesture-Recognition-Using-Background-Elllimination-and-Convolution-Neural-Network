from HGR import HGR
import cv2
import numpy as np

hgr = HGR()

show_statistics = False

while True:
    hgr.predict()

    cv2.imshow("Video feed", hgr.get_frame())

    if hgr.thresholded is not None:
        cv2.imshow("Thresholded", hgr.get_thresholded_hand())

    if show_statistics and hgr.predicted_class is not None:
        textImage = np.zeros((300, 512, 3), np.uint8)
        cv2.putText(textImage, "Pedicted Class : " + hgr.predicted_class,
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)

        cv2.putText(textImage, "Confidence : " + str(hgr.confidence * 100) + '%',
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)
        cv2.imshow("Statistics", textImage)

    # observe the keypress by the user
    keypress = cv2.waitKey(1) & 0xFF

    if keypress == ord('s'):
        show_statistics = True

    if keypress == ord('q'):
        break
