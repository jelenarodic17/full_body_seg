from pathlib import Path

from PIL import Image
import skimage.io as io
import numpy as np
import os
import skimage.filters as skfil
from skimage.io import imread

import matplotlib.pyplot as plt
from pycocotools.coco import COCO


# Putanje do slika i njihovih maski
IMAGE_DIR = os.path.normpath(r'C:\\Users\psiml\Downloads\train2017\train2017\\')
ANNOT_DIR = os.path.normpath(r'C:\Users\psiml\Downloads\annotations_trainval2017\annotations')
IMAGE_DIR_SAVE = os.path.normpath(r'C:\Users\psiml\Downloads\coco_train\images')
MASK_DIR_SAVE = os.path.normpath(r'C:\Users\psiml\Downloads\coco_train\masks')

mask_annot_path = r'C:\Users\psiml\Downloads\annotations_trainval2017\annotations\instances_train2017.json'


def make_masks_from_ann(origin_path, ann_path, save_path_mask, save_path_img):
    coco = COCO(ann_path)

    catIds = coco.getCatIds(catNms=['person'])  # izdvoji indekse date kategorije osoba
    imgIds = coco.getImgIds(catIds=catIds);  # daje imgIds iz date kategorije
    img = coco.loadImgs(ids=imgIds)
    img_ids = [dict['id'] for dict in img]
    """
    for id in imgIds:
        img = coco.loadImgs(ids=id)[0]
        I = Image.open(origin_path + '\\'+img['file_name'])
        I.save(save_path_img + '\\' + str(img['id']) + '.jpg')
    """
    # plt.imshow(I); plt.axis('off'); plt.show()
    annIds = coco.getAnnIds(imgIds=img_ids, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    #coco.showAnns(anns)
    # plt.imshow(I); plt.axis('off'); plt.show()
    for ann in anns:
        mask = coco.annToMask(ann)
        mask = mask*255
        #io.imshow(mask)
        mask = Image.fromarray(mask).convert('L')
        #plt.imshow(mask)
        mask.save(save_path_mask + '\\' + str(ann['image_id']) + '.png')
        # plt.imshow(mask, cmap='gray')
        # plt.axis('off')
        # plt.show()
        # I = Image.open(MASK_DIR_SAVE + '\\'+'.jpg')
        # io.imsave(save_path_mask + '\\' + str(ann['id']) + '.jpg',mask)


if __name__ == "__main__":
    make_masks_from_ann(origin_path=IMAGE_DIR, ann_path=mask_annot_path, save_path_mask=MASK_DIR_SAVE,
                        save_path_img=IMAGE_DIR_SAVE)
    #I = Image.open(MASK_DIR_SAVE + '\\' + '900100400851.jpg')
    #plt.imshow(I, cmap='gray')
    #plt.show()