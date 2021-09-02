# 以图识物模型

## 脚本运行说明

以图识物模型脚本为`infer_main.py`，可通过一下命令查看帮助信息：

```bash
python infer_main.py -h
```

输出：

```
Usage: infer_main.py [OPTIONS] PATH_IMG

  以图识物模型推断

  dir_img: 待预测的图片或图片的目录

Options:
  -o, --out_dir PATH  图像处理结果保存目录
  -h, --help          Show this message and exit.

```

## 示例

```buildoutcfg
python infer_main.py /home/nwom/image
```