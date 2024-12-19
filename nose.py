import cv2
import numpy as np
import tensorflow as tf

VID = cv2.VideoCapture(0)
MODEL = tf.keras.models.load_model("C:\\Users\\ernes\\OneDrive\\102\\p2\\4\keras_model.h5")

while True:
    ret, frame = VID.read()
    img = cv2.resize(frame,(224,224))
    tst = np.array(img, dtype=np.float32)
    tst2 = np.expand_dims(tst, axis=0)
    normal = tst2/255
    prediction = MODEL.predict(normal)
    print(tst, tst2, normal)
    print("Predicción: ", prediction)
    if not ret:
        print("ERROR")
        break
    cv2.imshow("", frame)
    key = cv2.waitKey(1)
    if key == 32:
        break

cv2.destroyAllWindows()                                                                                                 