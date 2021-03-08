# Initialising Modules
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np
from flask import *
import cv2
import base64

# Loading the model
model = load_model('Mask_Detector.model')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Starting the camera
labels_dict = {0: 'MASK', 1: 'NO MASK'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}


def compute(fm):
    color = cv2.cvtColor(fm, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(color, 1.3, 5)

    # Defining the box around the face
    for (x, y, w, h) in faces:
        face_img = color[y:y + w, x:x + h]
        resized = cv2.resize(face_img, (224, 224))
        img_arr = img_to_array(resized)
        pre_process = preprocess_input(img_arr)
        reshaped = np.reshape(pre_process, (1, 224, 224, 3))

        # Predict using model
        result = model.predict(reshaped)

        labels = np.argmax(result, axis=1)[0]
        percentage = np.round(np.max(result, axis=1) * 100, 2)

        cv2.rectangle(fm, (x, y), (x + w, y + h), color_dict[labels], 2)
        cv2.rectangle(fm, (x, y - 40), (x + w, y), color_dict[labels], -1)
        cv2.putText(fm, labels_dict[labels], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(fm, str(percentage), (x + 130, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return fm


app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/post', methods=['POST'])
def upload_image():
    filestr = request.files["file"].read()
    npimg = np.fromstring(filestr, numpy.uint8)
    compute(npimg)
    _, response_image = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(response_image)
    return {
        "image": jpg_as_text.decode('utf-8'),
        "hasMask": False,
        "confidence": 0.5
    }


if __name__ == "__main__":
    print("Hello World")
    app.run(debug=True)
