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

# Helper function to draw bounding boxes on the image
def draw_boxes(image, results):
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Class indices

        for box, conf, cls in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

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
    try:
        image_file = request.files["image"]
        image = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": e}), 400

    # Run YOLO inference
    try:
        results = model(image)
    except Exception as e:
        return jsonify({"error": "No image file provided"}), 400

    # Extract predictions
    detections = []
    if not results:
        return jsonify({"error": "Nothing detected"}), 400
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), 
                                  result.boxes.conf.cpu().numpy(), 
                                  result.boxes.cls.cpu().numpy()):
            detections.append({
                "class": model.names[int(cls)],
                "confidence": float(conf),
                "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
            })

    # return the image with bounding boxes
    output_image = draw_boxes(image.copy(), results)

    return jsonify({
        "message": "Inference completed successfully",      # Retrun Success Message
        "detections": detections,                           # Return Dtections
        "orignal_image": encode_image(image),               # Return Orignal Image(encoded)
        "output_image": encode_image(output_image)          # return output image(encoded)
    }), 200

if __name__ == "__main__":
    app.run()