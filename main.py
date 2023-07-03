import os
from flask import Flask, request, jsonify, json, abort, redirect, url_for, render_template, send_file, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from werkzeug.utils import secure_filename
import torch, torchvision
import os, pickle ,random
import cv2 as cv
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor
from io import BytesIO
from PIL import Image
import base64
import gcsfs
import numpy as np

cfg_save_path = "IS_cfg.pickle"

with open(cfg_save_path, "rb") as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
cfg.MODEL.DEVICE = "cpu"

project_id = os.environ.get("PROJECT_ID")
fs = gcsfs.GCSFileSystem(project=project_id)
bucket_name = os.environ.get("BUCKET_NAME")


predictor = DefaultPredictor(cfg)

def on_image(filepath, predictor, inputName):
    with fs.open(filepath, 'rb') as f:
        filename = f.read()
        f.close()
    image = np.asarray(bytearray(filename), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    outputs = predictor(image)
    b = (outputs["instances"].to("cpu"))
    if(len(b.scores.tolist()) > 0):
        confidence = b.scores.tolist()[0]
    else:
        confidence = 0
    v = Visualizer(image[:,:,::-1], metadata = {}, scale = 1, instance_mode = ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    data = v.get_image()
    img = Image.fromarray(data, 'RGB')
    with BytesIO() as output:
        img.save(output, format="JPEG")
        contents = output.getvalue()
    return confidence, contents
    

app = Flask(__name__)


@app.route('/upload/<filename>', methods=['PUT'])
def upload(filename):
    inputName = secure_filename(filename) 
    filepath = f"{bucket_name}/uploads/{inputName}"
    with fs.open(filepath, 'wb') as f:
        f.write(request.get_data())
        f.close()
    confidence, imageBytes = on_image(filepath, predictor, inputName)
    maskedImage = BytesIO()
    maskedImage.write(imageBytes)
    maskedImage.seek(0)
    response = make_response(send_file(maskedImage, mimetype='image/jpeg'))
    if(confidence != 0):
        response.headers['X-Confidence'] = 'damaged'
    elif(confidence == 0):
        response.headers['X-Confidence'] = 'undamaged'
    return response  


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))