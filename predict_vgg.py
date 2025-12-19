import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

IMAGE_SIZE = 224
MODEL_PATH = "models/vgg16_brain_tumor.h5"
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor", "no_tumor"]

model = load_model(MODEL_PATH)

def predict_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not load image.")
        return

    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_preprocessed = preprocess_input(img_resized.astype('float32'))
    img_input = np.expand_dims(img_preprocessed, axis=0)

    predictions = model.predict(img_input)[0]
    class_index = np.argmax(predictions)
    class_name = CLASS_NAMES[class_index]
    confidence = predictions[class_index] * 100

    print("----------------------------------")
    print("Predicted Class:", class_name)
    print("Confidence:", f"{confidence:.2f}%")
    print("----------------------------------")

    cv2.putText(img, f"{class_name} ({confidence:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# CHANGE THIS TO TEST A NEW IMAGE
predict_image("data/test/glioma_tumor/image(1).jpg")
