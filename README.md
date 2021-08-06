# Full body segmentation

Goal: to segment persons from an image 


Method: Unet model with pretrained encoder (vgg11) = ternaus net


Relevant literature: https://arxiv.org/pdf/1801.05746.pdf


Datasets: 

          https://www.kaggle.com/tapakah68/supervisely-filtered-segmentation-person-dataset/code

          https://cocodataset.org/#home
          
          
Results: shown in folders for network trained on coco dataset, and also on supervisely dataset. 

         IoU loss on test on supervisely: 0.8
         
         IoU loss on test on coco dataset: 0.5
