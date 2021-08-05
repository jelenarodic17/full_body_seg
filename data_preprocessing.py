from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms.functional as TF
from torch.nn import BCEWithLogitsLoss
import torch.optim as optim

import time
import copy
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path
import skimage.filters as skfil
from skimage.io import imread
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from unet_model import UNet11

# Ako na uredjaju postoji GPU koristicemo ga za obucavanje
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Putanje do slika i njihovih maski
IMAGE_DIR = os.path.normpath(r'C:\\Users\psiml\Downloads\supervisely_person_clean_2667_img\images')
MASK_DIR = os.path.normpath(r'C:\\Users\psiml\Downloads\supervisely_person_clean_2667_img\masks')
# Dimenzija na koju cemo smanjiti slike
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
# Velicina batch-a
BATCH_SIZE = 32

# Praznjenje kes memorije pre pocetka programa
torch.cuda.empty_cache()

# Uzimanje imena svih slika i maski iz navedenih direktorijuma
path_img = Path(IMAGE_DIR)
img_list = list(path_img.glob('*.png'))
path_mask = Path(MASK_DIR)
mask_list = list(path_mask.glob('*.png'))


class FullBodySegmentation(data.Dataset):
    def __init__(self, inputs: list, targets: list, transform=None):
        super().__init__()
        self.inputs = inputs # imena svih slika iz odredjenog skupa
        self.targets = targets # imena svih maski iz odredjenog skupa
        self.transform = transform

    def __len__(self, ):
        return len(self.inputs)

    # vraca jednu sliku i jednu masku na osnovu imena svih slika i indeksa koji je dobio
    def __getitem__(self, idx: int):
        input_image = self.inputs[idx]
        target_image = self.targets[idx]

        # Ucitavamo sliku na osnovu njenog imena i vrsimo permutaciju osa da odgovara tenzorskoj predstavi koja je
        # (3,height,width) za ulaz u resize funkciju, a zatim je normalizujemo i prebacujemo u tenzor
        image = np.moveaxis(np.array(Image.open(input_image).convert("RGB")),[2,0],[0,1])
        image = torch.tensor(image/(image.flatten().max()), dtype=torch.float32)
        mask = np.array(Image.open(target_image).convert("L"))
        # Za masku radimo slicnu stvar, sa razlikom sto nju podesavamo na dimenziju (1,height,width)
        mask =torch.tensor(mask / (mask.flatten().max()), dtype=torch.float32)
        mask = mask.reshape((1,mask.shape[0],mask.shape[1]))
        #print('ulaz image ' + str(image.size()) + ' ulaz mask ' + str(mask.size()))
        # Primenjujemo transformacije slike i maske
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        #print('image '+str(image.size())+' mask '+str(mask.size()))
        #print(r'\n')

        return image, mask

# Posto poziv klase Compose ne podrazumeva razlicite funkcije nad 2 slike (jednu 3D, a jednu 2D) izvedena je potklasa
# koja moze da resi ovaj problem, tj. podrazumeva da ima transformacije ulaza i transformacije izlaza
class ComposeNew(transforms.Compose):
    def __init__(self, transforms):
        super().__init__(transforms)
        self.transforms_input = transforms['input']
        self.transforms_output = transforms['output']

    def __call__(self, image, mask):
        for t in self.transforms_input:
            image = t(image)
        for t in self.transforms_output:
            mask = t(mask)
        return image, mask

# Transformacije koje prosledjujemo FullBodySegmentation klasi
train_transform = ComposeNew({
    'input': [
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomVerticalFlip(p=0.1),
        transforms.Normalize([0,0,0],[1,1,1]),
            ],
    'output': [
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.Normalize(0,1),
        #transforms.ToTensor(),
    ]}
)

val_transform = ComposeNew({
    'input': [
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.Normalize([0,0,0],[1,1,1]),
        #transforms.ToTensor(),
            ],
    'output': [
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.Normalize(0,1)
        #transforms.ToTensor(),
    ]}
)

# Delimo skup imena svih slika na test i ostalo
x_data, x_test, y_data, y_test = train_test_split(img_list, mask_list, test_size=0.1, random_state=42, shuffle=True)
# Delimo skup imena slika iz ostalo na train i val
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=42, shuffle=True)

# Loaders uzimaju kao atribute train_dir i train_maskdir sto su ustvari liste imena slika iz adekvatnog skupa
# To se prosledjuje FullBodySegmentation klasi koja dozvoljava otvaranje 1 po 1 slike, kada ih pozove DataLoader
def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, test_dir, test_maskdir, batch_size, train_transform,
                val_transform, num_workers=4, pin_memory=True):

    train_ds = FullBodySegmentation(inputs=train_dir, targets=train_maskdir,transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                              shuffle=True)

    val_ds = FullBodySegmentation(inputs=val_dir, targets=val_maskdir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    test_ds = FullBodySegmentation(inputs=test_dir, targets=test_maskdir, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader, test_loader

# Napravimo adekvatne loader-e
train_loader, val_loader, test_loader = get_loaders(x_train, y_train, x_val, y_val, x_test, y_test, BATCH_SIZE,
                                                    train_transform, val_transform, num_workers=2, pin_memory=True)


def train_model(model, criterion, optimizer, loaders, num_epochs=10):
    start_time = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())  # state_dict je recnik svih tezina zgodno za cuvanje
    best_bce = 1000.0
    loss_train_list = list()
    loss_val_list = list()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        loss_sum_train = list()
        loss_sum_val = list()

        # Epoha ima trening i validacionu fazu
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Model ide u trening mod
            else:
                model.eval()  # Model ide u evaluation mod

            k = 0
            #Iteracije kroz batch-eve
            for images, masks in loaders[phase]:
                print(k)
                k = k+1
                # Prebacujemo slike i maske na GPU
                images= images.to(device)
                masks = masks.to(device)

                # Obezbedjujemo se da su gradijenti vraceni na 0
                optimizer.zero_grad()

                # forward
                # uključuje računanje gradijenta u train fazi
                with torch.set_grad_enabled(phase == 'train'):
                    prediction_masks = model(images)
                    loss = criterion(prediction_masks, masks)

                    # backward, ako smo u fazi treniranja
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        loss_sum_train.append(float(loss))
                    if phase=='val':
                        loss_sum_val.append(float(loss))


            # deep copy the model, uzmi najbolji model sa validacije
            if phase == 'val' and loss < best_bce:
                best_bce = loss
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
        loss_train = np.array(loss_sum_train).mean()
        loss_train_list.append(loss_train)
        loss_val = np.array(loss_sum_val).mean()
        loss_val_list.append(loss_val)
        print('train loss: ' + str(loss_train) + ' val loss: ' + str(loss_val))


    time_elapsed = time.time() - start_time
    print(f'Trening trajao {(time_elapsed // 60):.0f}m {time_elapsed % 60:.0f}s')
    print('Best val loss: '+str(best_bce))

    # ucitavamo najbolji model u memoriju
    model.load_state_dict(best_model_wts)

    return model, loss_train_list, loss_val_list

def train_UNet11():

    loaders = {"train": train_loader, "val": val_loader}

    print(f'Broj slika u trening skupu: {len(train_loader) * BATCH_SIZE} images')
    print(f'Broj slika u validacionom skupu: {len(val_loader) * BATCH_SIZE} images')
    print(f'Broj slika u test skupu: {len(test_loader) * BATCH_SIZE} images')

    # Formiranje inastance modela i definisanje parametara za treniranje
    model = UNet11().to(device)

    optimizer = torch.optim.Adam(model.parameters())
    
    # criterion = {
    #    "dice": DiceLoss(),
    #    "iou": IoULoss(),
    #    "bce": BCEWithLogitsLoss()
    # }
    
    criterion = BCEWithLogitsLoss()
    # TRENIRANJE MODELA
    num_epochs = 10
    model,loss_train_list, loss_val_list = train_model(model, criterion, optimizer, loaders, num_epochs=num_epochs)


    # Cuvanje obucenog modela
    torch.save(model.state_dict(),
               os.path.normpath(r'C:\\Users\psiml\PycharmProjects\\ternaus3.pth'))

    plt.plot(np.arange(num_epochs),np.array(loss_val_list))
    plt.plot(np.arange(num_epochs),np.array(loss_train_list))
    plt.xlabel('redni broj epohe')
    plt.ylabel('loss')
    plt.legend(['val loss','train loss'])
    plt.show()

def loss_on_test(model,loader):
    criterion = BCEWithLogitsLoss()
    with torch.no_grad():
        loss = 0
        criterion_bce = torch.nn.BCEWithLogitsLoss()
        k = 0
        loss_vec = list()
        for batch in loader:
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)
            if len(images)==BATCH_SIZE:
                dim = BATCH_SIZE
            else:
                dim = len(images)
            masks = masks.reshape((dim, IMAGE_HEIGHT, IMAGE_WIDTH))
            batch_preds = torch.sigmoid(model(images))
            batch_preds = batch_preds.detach().reshape((dim, IMAGE_HEIGHT, IMAGE_WIDTH))
            loss_bce = criterion(batch_preds, masks)
            loss_vec.append(float(loss_bce))
            k = k+1
            print(k)
        loss_test = np.array(loss_vec).mean()
        print("--------------------")
        print('loss test je '+str(loss_test))

def test_UNet11():
    # TESTIRANJE I PRIKAZ REZULTATA
    # Prebacujemo model u mod za evaluaciju, da ne racuna gradijente
    model = UNet11()
    model.load_state_dict(torch.load(os.path.normpath(r'C:\Users\psiml\PycharmProjects\\ternaus2.pth')))
    model = model.to(device)
    model.eval()

    # Racunanje greske na test skupu
    print(f'Broj slika u test skupu: {len(test_loader) * BATCH_SIZE} images')
    loss_on_test(model,test_loader)

    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    axs = axs.ravel()
    j = 0
    for batch in test_loader:
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device)
        if len(images) == BATCH_SIZE:
            dim = BATCH_SIZE
        else:
            dim = len(images)
        masks = masks.reshape((dim, IMAGE_HEIGHT, IMAGE_WIDTH))
        batch_preds = torch.sigmoid(model(images))
        batch_preds = batch_preds.detach()
        for i in range(BATCH_SIZE):
            axs[3].imshow(batch_preds[i].cpu().squeeze(0).numpy(), cmap='gray')
            axs[0].imshow(images[i].cpu().permute(1, 2, 0).numpy())
            axs[1].imshow(masks[i].cpu().numpy(), cmap='gray')
            axs[2].imshow(implement_mask(batch_preds[i].cpu().squeeze(0), images[i].cpu()).permute(1, 2, 0))
            plt.savefig(
                r'C:\\Users\psiml\PycharmProjects\psiml_body_segmentation\Rezultati\Rezultati 2\\rez' + str(i) + str(
                    j) + '.png')
        j = j + 1


def implement_mask(mask, image):
    mask_3d = mask.repeat(3, 1, 1)
    return mask_3d * image


def median_filter(image):
    image1 = image.numpy()
    filtered_image = skfil.median(image.numpy(), np.ones((5, 5)))
    return filtered_image



if __name__=='__main__':
    #train_UNet11()
    test_UNet11()
