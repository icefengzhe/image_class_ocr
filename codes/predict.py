import torch
import requests
import sys
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from .utils import img2base64

from ecloud import CMSSEcloudOcrClient
import json

transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])


def predict(image_path, model, device, ckpt, labels):
    image = Image.open(image_path)
    image = transform(image)
    image = torch.unsqueeze(image, dim=0)
    print(image.shape)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    with torch.no_grad():
        output = model(image)
        pred = F.softmax(output)
        pred = torch.argmax(pred)
    pred = labels[pred]
    return pred


def ocr(image_path):
    image_base64 = img2base64(image_path)
    print(sys.getsizeof(image_base64))
    url = 'https://api-wuxi-1.cmecloud.cn:8443'
    requesturl = '/api/ocr/v1/webimage'

    accesskey = "01e3bd25acd74fb7beed36648c7140fb"
    secretkey = "7e2d1a1c7cc64ff588e22af73941bb10"
    try:
        ocr_client = CMSSEcloudOcrClient(accesskey, secretkey, url)
        response = ocr_client.request_ocr_service_file(requestpath=requesturl, imagepath=image_path)
        words = json.loads(response.text)['body']['content']['prism_wordsInfo']
        string = '#'.join([item['word'] for item in words])
        return string
    except ValueError as e:
        print(e)


if __name__ == '__main__':
    # import torch.nn as nn
    #
    # DEVICE = torch.device('cpu')
    # labels = ['电杆', '光交箱', '人手井']
    # model = torchvision.models.resnet50(pretrained=True)
    # model.fc = nn.Sequential(
    #     nn.Linear(2048, len(labels))
    # )
    # pred = predict('./0d761b8cf3bf43338b6d04d2f4230815.jpg', model, DEVICE, '../models_1/ckpt_e40_acc0.9717076277934072.pth', labels)
    # print(pred)

    ocr('./0d761b8cf3bf43338b6d04d2f4230815.jpg')
