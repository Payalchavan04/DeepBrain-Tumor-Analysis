import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ---------- SETTINGS ----------
IMAGE_SIZE = 150
model_path = "models/brain_tumor_model.h5"

# ---------- LOAD MODEL ----------
model = load_model(model_path)

# Class labels in the same order as training folder
class_labels = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor", "no_tumor"]

def predict_image(image_path):
    # Load image
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not load image. Check the file path.")
        return

    # Resize image
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # Normalize (0–1)
    img_normalized = img_resized / 255.0

    # Expand dimension → model expects (1, 150, 150, 3)
    img_input = np.expand_dims(img_normalized, axis=0)

    # Predict
    prediction = model.predict(img_input)[0]

    # Get index of highest probability
    class_index = np.argmax(prediction)

    # Get class name
    class_name = class_labels[class_index]

    # Get confidence
    confidence = prediction[class_index] * 100

    print("-------------------------------------")
    print(f"Predicted Class: {class_name}")
    print(f"Confidence: {confidence:.2f}%")
    print("-------------------------------------")

    # Show image with predicted label
    cv2.putText(img, class_name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------- TEST THE FUNCTION --------
# Replace with your image path:
# Example: predict_image("data/test/glioma_tumor/Y1.jpg")

# Just change the path below:
predict_image("data/test/glioma_tumor/image(1).jpg")

