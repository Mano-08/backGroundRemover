import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation

cap = cv2.VideoCapture(0)

width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

background_image = cv2.resize(cv2.imread("surprise.jpg"), (width, height))

segmentor = SelfiSegmentation()

while True:
    ret, frame = cap.read()

    segmentated_img = segmentor.removeBG(
        frame, background_image, threshold=0.9)

    concatenated_img = cv2.hconcat([frame, segmentated_img])

    cv2.imshow("Camera Live", concatenated_img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
