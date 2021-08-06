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

    for img_id in imgIds: # prolazimo kroz sve slike ljudi
        img = coco.loadImgs(ids=img_id)[0]
        I = np.array(Image.open(origin_path + '\\'+img['file_name']).convert('RGB'))
        # I.save(save_path_img + '\\' + str(img['id']) + '.jpg')
        # Ucitavamo anotacije za pojedinacnu sliku
        annIds = coco.getAnnIds(imgIds=img_id, catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        final_mask = np.zeros((np.shape(I)[0],np.shape(I)[1]))
        for ann in anns: # Vrsimo spajanje svih maski za vise ljudi u jednu
            mask = coco.annToMask(ann)
            mask = mask*255
            final_mask = final_mask + mask
        final_mask[final_mask>255]=255
        #plt.imshow(final_mask)
        final_mask = Image.fromarray(final_mask).convert('L')
        final_mask.save(save_path_mask + '\\' + str(ann['image_id']) + '.png')



if __name__ == "__main__":
    make_masks_from_ann(origin_path=IMAGE_DIR, ann_path=mask_annot_path, save_path_mask=MASK_DIR_SAVE,
                        save_path_img=IMAGE_DIR_SAVE)
