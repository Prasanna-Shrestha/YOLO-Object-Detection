# Module import
from flask import Flask, render_template, request
# from werkzeug.utils import secure_filename
import torch
import cloudinary
import cloudinary.uploader
import io
from PIL import Image
import uuid
import os

app = Flask(__name__, static_url_path='/static')

cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET'),
    secure=True
)


# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
model.conf = 0.45

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
        # Not necessary to sanitize file name 
        # as the file is being directly uploaded to cloudinary and
        # not being stored in local device
        # filename = secure_filename(file.filename)

        # Upload image to the cloudinary
        upload_img_uuid = f"yolo_upload{uuid.uuid4().hex[:8]}"
        upload_img = cloudinary.uploader.upload(file, public_id=upload_img_uuid, folder="yolo_uploads")

        # Get the image URL
        image_url = upload_img['secure_url']
        results = model(image_url)
        results.render()

        # Convert to image and save to memory
        img_array = results.ims[0]
        img_pil = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG")
        buffer.seek(0)

        # Detect objects in the image
        result_img_uuid = f"yolo_result_{uuid.uuid4().hex[:8]}"
        # resource_type needs to be set to "image" as the buffer is being uploaded that has no fixed extension unlike the previous case
        upload_result = cloudinary.uploader.upload(buffer, resource_type="image", public_id=result_img_uuid, folder="yolo_result")
        result_url = upload_result['secure_url']

        # Parse detections
        detections = results.pandas().xyxy[0][['name', 'confidence']].to_dict(orient='records')

        return render_template('upload.html',
                               image_url= image_url,
                               objects=detections,
                               detected_img= result_url)

if __name__ == '__main__':
    # for local deployment
    # app.run(debug=True)

    # for production deployment
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
