from flask import Flask, jsonify, request
from PIL import Image
from ultralytics import YOLO
from flask_cors import CORS
import io

# Load YOLOv8 model
model = YOLO("model/v1.pt")  # ganti dengan path model Anda
names = model.names

app = Flask(__name__)
CORS(app)



def classify(image):
    results = model(image)
    
    objects = []
    for result in results:
        boxes = result.boxes  
        for c in boxes.cls:
            label = names[int(c)]
            if label not in objects:
                objects.append(label)
    
    return objects

@app.route('/')
def index():
    return 'HeyFo API'

@app.route('/predict', methods=['POST'])
def predict():
    if request.files['image']:
        try:
            image_file = request.files["image"]
            image_bytes = image_file.read()
            img = Image.open(io.BytesIO(image_bytes))
            objects = classify(img)
            
            return jsonify(
                {
                    'status' : 'success',
                    'objects': objects
                    }
                )
        except Exception as e:
            return jsonify({
                'status': 'failed',
                'error': str(e)
                })

@app.route('/get-data', methods=['GET'])
def getData():
    return jsonify({
        'data': names
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
