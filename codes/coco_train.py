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
import torch.nn.functional as F
import skimage.io as io

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
IMAGE_DIR = os.path.normpath(r'C:\Users\psiml\Downloads\coco_train\images')
MASK_DIR = os.path.normpath(r'C:\Users\psiml\Downloads\coco_train\masks')
IMAGE_DIR_TEST = os.path.normpath(r'C:\Users\psiml\Downloads\coco_val\images') # namerno val jer u testu nemamo labela
MASK_DIR_TEST = os.path.normpath(r'C:\Users\psiml\Downloads\coco_val\images')
# Dimenzija na koju cemo smanjiti slike
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
# Velicina batch-a
BATCH_SIZE = 50

# Praznjenje kes memorije pre pocetka programa
torch.cuda.empty_cache()

# Uzimanje imena svih slika i maski iz navedenih direktorijuma
path_img = Path(IMAGE_DIR)
img_list = list(path_img.glob('*.jpg'))
path_mask = Path(MASK_DIR)
mask_list = list(path_mask.glob('*.png'))
path_img_test = Path(IMAGE_DIR_TEST)
img_list_test = list(path_img_test.glob('*.jpg'))
path_mask_test = Path(MASK_DIR_TEST)
mask_list_test = list(path_mask_test.glob('*.png'))

class FullBodySegmentation(data.Dataset):
    def __init__(self, inputs: list, targets: list, transform=None):
        super().__init__()
        self.inputs = inputs  # imena svih slika iz odredjenog skupa
        self.targets = targets  # imena svih maski iz odredjenog skupa
        self.transform = transform

    def __len__(self, ):
        return len(self.inputs)

    # vraca jednu sliku i jednu masku na osnovu imena svih slika i indeksa koji je dobio
    def __getitem__(self, idx: int):
        input_image = self.inputs[idx]
        target_image = self.targets[idx]

        # Ucitavamo sliku na osnovu njenog imena i vrsimo permutaciju osa da odgovara tenzorskoj predstavi koja je
        # (3,height,width) za ulaz u resize funkciju, a zatim je normalizujemo i prebacujemo u tenzor
        image = np.moveaxis(np.array(Image.open(input_image).convert("RGB")), [2, 0], [0, 1])
        image = torch.tensor(image / (image.flatten().max()), dtype=torch.float32)
        mask = np.array(Image.open(target_image).convert("L"))
        # Za masku radimo slicnu stvar, sa razlikom sto nju podesavamo na dimenziju (1,height,width)
        mask = torch.tensor(mask / (mask.flatten().max()), dtype=torch.float32)
        mask = mask.reshape((1, mask.shape[0], mask.shape[1]))
        # print('ulaz image ' + str(image.size()) + ' ulaz mask ' + str(mask.size()))
        # Primenjujemo transformacije slike i maske
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        # print('image '+str(image.size())+' mask '+str(mask.size()))
        # print(r'\n')

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
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.1),
        transforms.Normalize([0, 0, 0], [1, 1, 1]),
    ],
    'output': [
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.Normalize(0, 1),
        # transforms.ToTensor(),
    ]}
)

val_transform = ComposeNew({
    'input': [
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.Normalize([0, 0, 0], [1, 1, 1]),
        # transforms.ToTensor(),
    ],
    'output': [
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.Normalize(0, 1)
        # transforms.ToTensor(),
    ]}
)

# Delimo skup imena svih slika na test i ostalo
#x_data, x_test, y_data, y_test = train_test_split(img_list, mask_list, test_size=0.1, random_state=42, shuffle=True)
# Delimo skup imena slika iz ostalo na train i val
x_train, x_val, y_train, y_val = train_test_split(img_list, mask_list, test_size=0.1, random_state=42, shuffle=True)

# Loaders uzimaju kao atribute train_dir i train_maskdir sto su ustvari liste imena slika iz adekvatnog skupa
# To se prosledjuje FullBodySegmentation klasi koja dozvoljava otvaranje 1 po 1 slike, kada ih pozove DataLoader
def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, test_dir, test_maskdir, batch_size, train_transform,
                val_transform, num_workers=4, pin_memory=True):
    train_ds = FullBodySegmentation(inputs=train_dir, targets=train_maskdir, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                              shuffle=True)

    val_ds = FullBodySegmentation(inputs=val_dir, targets=val_maskdir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                            shuffle=False)

    test_ds = FullBodySegmentation(inputs=test_dir, targets=test_maskdir, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                             shuffle=False)

    return train_loader, val_loader, test_loader


# Napravimo adekvatne loader-e
train_loader, val_loader, test_loader = get_loaders(x_train, y_train, x_val, y_val, img_list_test, mask_list_test, BATCH_SIZE,
                                                    train_transform, val_transform, num_workers=2, pin_memory=True)


def train_model(model, criterion, optimizer, loaders, num_epochs=10):
    start_time = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())  # state_dict je recnik svih tezina zgodno za cuvanje
    best_bce = 1000.0
    loss_train_list = list()
    loss_val_list = list()
    iou_train_list = list()
    iou_val_list = list()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        loss_sum_train = list()
        loss_sum_val = list()
        iou_sum_train = list()
        iou_sum_val = list()

        # Epoha ima trening i validacionu fazu
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Model ide u trening mod
            else:
                model.eval()  # Model ide u evaluation mod

            k = 0
            # Iteracije kroz batch-eve
            for images, masks in loaders[phase]:
                print(k)
                k = k + 1
                # Prebacujemo slike i maske na GPU
                images = images.to(device)
                masks = masks.to(device)

                # Obezbedjujemo se da su gradijenti vraceni na 0
                optimizer.zero_grad()

                # forward
                # uklju??uje ra??unanje gradijenta u train fazi
                with torch.set_grad_enabled(phase == 'train'):
                    prediction_masks = model(images)
                    loss = criterion(prediction_masks, masks)

                    # backward, ako smo u fazi treniranja
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        prediction_masks = prediction_masks.detach()
                        loss_sum_train.append(float(loss))
                        masks = masks.detach()
                        iou_loss_pom = iou_loss(torch.sigmoid(prediction_masks), masks).detach()
                        iou_sum_train.append(iou_loss_pom.to('cpu'))
                    if phase == 'val':
                        prediction_masks = prediction_masks.detach()
                        loss_sum_val.append(float(loss))
                        masks = masks.detach()
                        iou_loss_pom = iou_loss(torch.sigmoid(prediction_masks), masks).detach()
                        iou_sum_val.append(iou_loss_pom.to('cpu'))

        print()
        loss_train = np.array(loss_sum_train).mean()
        loss_train_list.append(loss_train)
        loss_val = np.array(loss_sum_val).mean()
        loss_val_list.append(loss_val)
        print('train loss: ' + str(loss_train) + ' val loss: ' + str(loss_val))

        iou_train = np.array(iou_sum_train).mean()
        iou_train_list.append(iou_train)
        iou_val = np.array(iou_sum_val).mean()
        iou_val_list.append(iou_val)
        print('train iou: ' + str(iou_train) + ' val iou: ' + str(iou_val))

        # deep copy the model, uzmi najbolji model sa validacije
        if loss_val < best_bce:
            best_bce = loss_val
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - start_time
    print(f'Trening trajao {(time_elapsed // 60):.0f}m {time_elapsed % 60:.0f}s')
    print('Best val loss: ' + str(best_bce))

    # ucitavamo najbolji model u memoriju
    model.load_state_dict(best_model_wts)

    return model, loss_train_list, loss_val_list, iou_train_list, iou_val_list


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
    num_epochs = 8
    model, loss_train_list, loss_val_list, iou_train, iou_val = train_model(model, criterion, optimizer, loaders,
                                                                            num_epochs=num_epochs)

    # Cuvanje obucenog modela
    torch.save(model.state_dict(),
               os.path.normpath(r'C:\\Users\psiml\PycharmProjects\\ternaus_coco2.pth'))

    plt.plot(np.arange(num_epochs), np.array(loss_val_list))
    plt.plot(np.arange(num_epochs), np.array(loss_train_list))
    plt.plot(np.arange(num_epochs), np.array(iou_train))
    plt.plot(np.arange(num_epochs), np.array(iou_val))
    plt.xlabel('redni broj epohe')
    plt.ylabel('loss, iou')
    plt.legend(['val loss', 'train loss','val iou', 'train iou'])
    plt.savefig('train_val_loss_iou.png')


def iou_loss(tensor1, tensor2):
    maximum = torch.max(tensor1.flatten())
    if maximum > 1:
        print("verovatnoca nije dobra:", maximum)
    intersection = tensor1 * tensor2
    union = torch.clamp((tensor1 + tensor2), min=0, max=1)
    intersection = torch.sum(torch.flatten(intersection, 1), dim=1)
    union = torch.sum(torch.flatten(union, 1), dim=1)
    return torch.mean(intersection / union)


def loss_on_test(model, loader):
    criterion = BCEWithLogitsLoss()
    with torch.no_grad():
        loss = 0
        k = 0
        loss_vec = list()
        iou_interpret = list()
        for batch in loader:
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)
            if len(images) == BATCH_SIZE:
                dim = BATCH_SIZE
            else:
                dim = len(images)
            masks = masks.reshape((dim, IMAGE_HEIGHT, IMAGE_WIDTH))
            batch_preds = torch.sigmoid(model(images))
            batch_preds = batch_preds.detach().reshape((dim, IMAGE_HEIGHT, IMAGE_WIDTH))
            # loss_bce = criterion(batch_preds, masks)
            # loss_vec.append(float(loss_bce))

            iou_interpret.append(iou_loss(batch_preds, masks).to('cpu'))
            k = k + 1
            print(k)
        # loss_test = np.array(loss_vec).mean()
        iou_test = np.mean(iou_interpret)
        print("--------------------")
        # print('loss test je '+str(loss_test))
        print('iou na testu je ' + str(iou_test))


def test_UNet11():
    # TESTIRANJE I PRIKAZ REZULTATA
    # Prebacujemo model u mod za evaluaciju, da ne racuna gradijente
    model = UNet11()
    model.load_state_dict(torch.load(os.path.normpath(r'C:\Users\psiml\PycharmProjects\\ternaus3.pth')))
    model = model.to(device)
    model.eval()

    # Racunanje greske na test skupu
    print(f'Broj slika u test skupu: {len(test_loader) * BATCH_SIZE} images')
    loss_on_test(model, test_loader)

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
    image1 = np.array(image)
    filtered_image = skfil.median(image.numpy(), np.ones((7, 7)))
    filtered_image[filtered_image > 0.3] = 1
    return torch.tensor(filtered_image)


def process_one_image(img_path, save_path):
    # Ucitavanje modela
    model = UNet11()
    model.load_state_dict(torch.load(os.path.normpath(r'C:\Users\psiml\PycharmProjects\\ternaus3.pth')))
    model = model.to(device)
    model.eval()

    # Ucitavanje zeljene slike i cuvanje originalnih dimenzija
    img = np.array(Image.open(img_path).convert("RGB"))
    img_shape = np.shape(img)

    # Priprema slike za transformacije
    image = np.moveaxis(img, [2, 0], [0, 1])
    image = torch.tensor(image / (image.flatten().max()), dtype=torch.float32)
    image = image.to(device)

    # Pravljenje dummy maske i reshape slike za ulaz u model
    mask = torch.zeros(1, IMAGE_HEIGHT, IMAGE_WIDTH)
    image, mask = train_transform(image, mask)
    image = image.reshape(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)

    # Vrsenje predikcije
    torch.set_grad_enabled(False)
    preds = torch.sigmoid(model(image))

    # Interpolacija na pocetnu dimenziju
    mask_resized = F.interpolate(preds, (img_shape[0], img_shape[1]))
    mask_resized = mask_resized.reshape(img_shape[0], img_shape[1])
    # mask_resized = median_filter(mask_resized.cpu())

    # Cuvanje rezultata
    fig, axs = plt.subplots(1, 2, figsize=(15, 15))
    axs = axs.ravel()
    axs[0].imshow(img)
    axs[1].imshow(np.array(
        implement_mask(mask_resized.squeeze(0).cpu(), np.moveaxis(img, [2, 0], [0, 1])).permute(1, 2, 0)) / np.max(img))
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    plt.savefig(save_path)


if __name__ == '__main__':
    train_UNet11()
    # test_UNet11()
    #img_path = r'C:\Users\psiml\Downloads\slike\moja_slika8.png'
    #save_path = r'C:\\Users\psiml\PycharmProjects\psiml_body_segmentation\Rezultati\Rezultati 2\\rez' + 'moja_slika8.png'
    #process_one_image(img_path, save_path)
