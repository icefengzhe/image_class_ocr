import click
import torch
import torchvision
import torch.nn as nn
import pandas as pd
from pathlib import Path
from codes.predict import predict, ocr

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('path_img', type=click.Path(exists=True))
@click.option('-o', '--out_dir', type=click.Path(), default=None, help='图像处理结果保存目录')
def main(path_img, out_dir):
    """以图识物模型推断

    \b
    dir_img: 待预测的图片或图片的目录
    """
    path_img = Path(path_img)

    if out_dir is None:
        out_dir = path_img.with_name(path_img.stem + '_out')
    else:
        out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    out_file = Path.joinpath(out_dir, 'result.csv')
    if not out_file.exists():
        result = pd.DataFrame(columns=['图片', '类别', '文字内容'])
        result.to_csv(out_file, encoding='gbk', index=False)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = './models_1/ckpt_e40_acc0.9717076277934072.pth'
    labels = ['电杆', '光交箱', '人手井']
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(2048, len(labels))
    )

    result = pd.DataFrame()
    if path_img.is_file():
        content = ocr(path_img)
        pred = predict(path_img, model, DEVICE, ckpt, labels)
        result = pd.concat([result, pd.DataFrame([[path_img, pred, content]])], axis=0)
    else:
        for img_path in path_img.glob('*.jpg'):
            content = ocr(img_path)
            pred = predict(img_path, model, DEVICE, ckpt, labels)
            result = pd.concat([result, pd.DataFrame([[img_path, pred, content]])], axis=0)
            print(result)
    result.to_csv(out_file, mode='a', encoding='gbk', header=False, index=False)


if __name__ == '__main__':
    main()
