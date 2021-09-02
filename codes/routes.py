import torch
import torchvision
import torch.nn as nn
import pandas as pd
import json
import shutil
from pathlib import Path
from flask import Flask, abort, request, jsonify, render_template
from .imagesimilary.image_similarity_main import image_is_repeat
from .predict import predict, ocr
from .db import result_to_db
from datetime import timedelta

app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = "application/json;charset=utf-8"

ALLOWED_EXTENSIONS = set(['jpg'])

database = '/home/fengzhe/image'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/img-similarity', methods=['POST', 'GET'])
def upload():
    if request.method == "POST":
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于jpg"})
        path_img = f'./tmp/{f.filename}'
        f.save(path_img)
        path_img = Path(path_img)

        repeat = []
        if path_img.is_file():
            if image_is_repeat(path_img, database):
                repeat.append(path_img)
            else:
                shutil.move(str(path_img), str(database))
        else:
            for img_path in path_img.glob('*.jpg'):
                if image_is_repeat(img_path, database):
                    repeat.append(str(img_path))
                else:
                    shutil.move(str(path_img), str(database))
        if repeat:
            # return jsonify({"error": 1001, "msg": "存在已解析过的图片"})
            return render_template('upload_error.html')
        else:
            return render_template('upload_ok.html')
    return render_template('upload.html')


@app.route('/img-recognize', methods=['POST', 'GET'])
def parse_image():
    # if not request.json:
    #     abort(400)
    if request.method == "POST":
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于jpg"})
        path_img = f'./tmp/{f.filename}'
        f.save(path_img)
        path_img = Path(path_img)

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = './models_1/ckpt_e40_acc0.9717076277934072.pth'
        labels = ['电杆', '光交箱', '人手井']
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(2048, len(labels))
        )

        repeat = []
        if path_img.is_file():
            if image_is_repeat(path_img, database):
                repeat.append(path_img)
            else:
                content = ocr(path_img)
                pred = predict(path_img, model, DEVICE, ckpt, labels)
                result = pd.DataFrame([[path_img, pred, content]], columns=['image', 'type', 'content'])
                result_to_db(result)
                shutil.move(str(path_img), str(database))
        else:
            for img_path in path_img.glob('*.jpg'):
                if image_is_repeat(img_path, database):
                    repeat.append(str(img_path))
                else:
                    content = ocr(img_path)
                    pred = predict(img_path, model, DEVICE, ckpt, labels)
                    result = pd.DataFrame([[img_path, pred, content]], columns=['image', 'type', 'content'])
                    result_to_db(result)
                    shutil.move(str(img_path), str(database))
        if repeat:
            # return jsonify({"error": 1001, "msg": "存在已解析过的图片"})
            return render_template('upload_error_1.html')
        else:
            return render_template('upload_ok_1.html')
    return render_template('upload_1.html')
