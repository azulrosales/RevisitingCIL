import numpy as np
import kagglehub
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from utils.toolkit import split_images_labels


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

def build_transform(is_train):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t

class iCIFAR224(iData):
    use_path = False
    
    train_trsf = build_transform(True)
    test_trsf = build_transform(False)
    common_trsf = [
        # transforms.ToTensor(),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
        print('self.train_data:', self.train_data)
        print('Type of self.train_data:', type(self.train_data))
        print('self.train_targets:', self.train_targets)
        print('Type of self.train_targets:', type(self.train_targets))

class FruitQuality(iData):
    use_path = False
    
    train_trsf = build_transform(True)
    test_trsf = build_transform(False)
    
    class ConvertToRGB:
        def __call__(self, img):
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        
    common_trsf = [
        ConvertToRGB()
    ]

    class_order = np.arange(28).tolist()

    def download_data(self):
        path = kagglehub.dataset_download("muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten") + '/Fruit And Vegetable Diseases Dataset'
        data = ImageFolder(root=path)

        train_dataset, test_dataset = random_split(data, [0.7, 0.3])

        self.train_data, self.train_targets = split_images_labels(train_dataset.dataset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dataset.dataset.imgs)
