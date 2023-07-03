import os
from flask import Flask, request, jsonify, json, abort, redirect, url_for, render_template, send_file, Response
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
import numpy as np

cfg_save_path = "IS_cfg.pickle"

with open(cfg_save_path, "rb") as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.895
cfg.MODEL.DEVICE = "cpu"

project_id = os.environ.get("PROJECT_ID")
bucket_name = os.environ.get("BUCKET_NAME")


predictor = DefaultPredictor(cfg)

def damaged(confidence):
    if(confidence > 0):
        return "DAMAGED"
    elif(confidence == 0):
        return "UNDAMAGED"

def on_image(image, predictor):
    image = np.asarray(bytearray(image), dtype="uint8")
    infImage = cv.imdecode(image, cv.IMREAD_COLOR)
    outputs = predictor(infImage)
    b = (outputs["instances"].to("cpu"))
    if(len(b.scores.tolist()) > 0):
        confidence = b.scores.tolist()[0]
    else:
        confidence = 0
    v = Visualizer(infImage[:,:,::-1], metadata = {}, scale = 1, instance_mode = ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    data = v.get_image()
    img = Image.fromarray(data, 'RGB')
    with BytesIO() as output:
        img.save(output, format="JPEG")
        contents = output.getvalue()
    return confidence, contents
    

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json(force=True)
    image = message['b64Image']
    bytes_img = encode(image, 'utf-8')
    binary_img = base64.decodebytes(bytes_img)
    confidence, imageBytes = on_image(binary_img, predictor)
    sevDamaged = damaged(confidence)
    b64Image = base64.b64encode(imageBytes)
    b64Dict = {"b64ImageWOverlay":f"{b64Image}","defects":{"severity":f"{sevDamaged}"}}
    return b64Dict


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))