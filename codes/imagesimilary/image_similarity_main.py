from .imagSimilary import *
import os
import shutil
from pathlib import Path

# 融合相似度阈值
threshold1 = 0.85
# 最终相似度较高判断阈值
threshold2 = 0.98


# 融合函数计算图片相似度
def calc_image_similarity(img1_path, img2_path):
    """
    :param img1_path: filepath+filename
    :param img2_path: filepath+filename
    :return: 图片最终相似度
    """

    similary_ORB = float(ORB_img_similarity(img1_path, img2_path))
    similary_phash = float(phash_img_similarity(img1_path, img2_path))
    similary_hist = float(calc_similar_by_path(img1_path, img2_path))
    # 如果三种算法的相似度最大的那个大于0.85，则相似度取最大，否则，取最小。
    max_three_similarity = max(similary_ORB, similary_phash, similary_hist)
    min_three_similarity = min(similary_ORB, similary_phash, similary_hist)
    if max_three_similarity > threshold1:
        result = max_three_similarity
    else:
        result = min_three_similarity

    return round(result, 3)


def image_is_repeat(image_path, database):
    database = Path(database)
    for image_db in database.glob('*.jpg'):
        kk = calc_image_similarity(image_path, image_db)
        if kk >= threshold2:
            print(image_db, kk)
            return True
    else:
        # shutil.move(str(image_path), str(database))
        return False


if __name__ == '__main__':

    # 搜索图片路径和文件名
    img1_path = 'F:/img_spam/data/train/unqrcode/10064003003550210800320010011888.jpg'

    # 搜索文件夹
    filepath = 'F:/img_spam/data/train/unqrcode/'

    # 相似图片存放路径
    newfilepath = 'F:/img_spam/4/第九组/'

    for parent, dirnames, filenames in os.walk(filepath):
        for filename in filenames:
            # print(filepath+filename)
            img2_path = filepath + filename
            kk = calc_image_similarity(img1_path, img2_path)

            try:
                if kk >= threshold2:
                    print(img2_path, kk)
                    shutil.copy(img2_path, newfilepath)
            except Exception as e:
                # print(e)
                pass
