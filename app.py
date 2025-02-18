from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load the YOLO model (supports .pt or .onnx models)
try:
    model = YOLO("./models/best.onnx")  # Replace with your model path (yolov8.pt for PyTorch)
except Exception as e:
    print('Error encountered: ', e)

# encode the image to send as jason in response
def encode_image(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

# Temp Home page to check if the webapp is deployed
@app.route('/')
def Home():
    return 'Home Page'

# Actual API to predtict using onix or pt
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Load image from request
    if not request.files["image"]:
        return jsonify({"error": "Missing value of the file"}), 403
    
    try:
        image_file = request.files["image"]
        image = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": e}), 403

    # Run YOLO inference
    try:
        print('Infering Image for results')
        results = model(image)
    except Exception as e:
        return jsonify({"error": f"unable to process image. {e}"}), 500
    
    detections = []
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), 
                                  result.boxes.conf.cpu().numpy(), 
                                  result.boxes.cls.cpu().numpy()):
            detections.append({
                "class": model.names[int(cls)],
                "confidence": float(conf),
                "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
            })

    return jsonify({
        "message": "Inference completed successfully",      # Retrun Success Message
        "detections": detections                           # Return Dtections
    }), 200

if __name__ == "__main__":
    app.run()