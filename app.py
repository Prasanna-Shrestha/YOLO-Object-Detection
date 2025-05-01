from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import imageio.v3 as iio
import torch
import glob
import shutil

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
DISPLAY_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLO model once
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part"
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        # Run detection - YOLO saves output to runs/detect/exp by default
        img = iio.imread(input_path)
        results = model(img)
        results.save()

        # Get path to the most recent YOLO output directory
        latest_exp = sorted(glob.glob('runs/detect/exp*'), key=os.path.getmtime, reverse=True)[0]
        # latest_exp = exp_dirs[0]
        output_img_path = glob.glob(os.path.join(latest_exp, '*.jpg'))[0]

        # Copy the result to static/outputs for web display
        display_path = os.path.join(DISPLAY_FOLDER, filename)
        shutil.copy(output_img_path, display_path)

        # Parse detections
        detections = results.pandas().xyxy[0][['name', 'confidence']].to_dict(orient='records')

        return render_template('upload.html',
                               image_url= input_path,
                               objects=detections,
                               detected_img='/' + display_path)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(DISPLAY_FOLDER, exist_ok=True)
    app.run(debug=True)
