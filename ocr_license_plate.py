# import the necessary packages
from pyimagesearch.anpr import PyImageSearchANPR
import imutils
import cv2

def cleanup_text(text):
    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

# initialize our ANPR class
anpr = PyImageSearchANPR(debug=False)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)

# loop over frames from the video stream
# loop over frames from the video stream
while True:
    # grab the current frame
    ret, frame = cap.read()

    # check if the frame was successfully captured
    if not ret:
        print("[INFO] No frame captured from the webcam. Exiting...")
        break

    # resize the frame and convert it to grayscale
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#	print("Frame shape:", frame.shape)

#	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# rest of your code...

    # apply automatic license plate recognition
    (lpText, lpCnt) = anpr.find_and_ocr(gray, psm=7,
        clearBorder=False)

    # only continue if the license plate was successfully OCR'd
    if lpText is not None and lpCnt is not None:
        # fit a rotated bounding box to the license plate contour and
        # draw the bounding box on the license plate
        box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
        box = box.astype("int")
        cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

        # compute a normal (unrotated) bounding box for the license
        # plate and then draw the OCR'd license plate text on the
        # image
        (x, y, w, h) = cv2.boundingRect(lpCnt)
        cv2.putText(frame, cleanup_text(lpText), (x, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # show the output ANPR image
    cv2.imshow("Output ANPR", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# cleanup
print("[INFO] cleaning up...")
cap.release()
cv2.destroyAllWindows()
