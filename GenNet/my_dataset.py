import os

import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image


class DataTransform:
    def __init__(self):#0.0039215689
        self.transforms = T.Compose([
            T.ToTensor(),
        ])

    def __call__(self, img, target):
        return self.transforms(img)*255, self.transforms(target)


def list_dir(root):
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    images = [int(i.split('.')[0]) for i in os.listdir(root)
              if os.path.splitext(i)[-1] in supported]
    ext = os.path.splitext(os.listdir(root)[0])[-1]
    images_sort = []
    for i in range(max(images)+1):
        img = '{}/{}{}'.format(root, i, ext)
        if os.path.exists(img):
            images_sort.append(img)
    return images_sort


class Dataset(data.Dataset):
    def __init__(self, root, transforms=None, subset='train'):
        super(Dataset, self).__init__()
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        if subset == "test":
            image_dir = root
            mask_dir = root
        else:
            image_dir = os.path.join(root, 'mask_space')
            mask_dir = os.path.join(root, 'mask_path')

        self.images = list_dir(image_dir)
        self.masks = list_dir(mask_dir)

        assert (subset in {'train', 'val', 'test'})

        if subset == 'train':
            self.images = self.images[:320000]
            self.masks = self.masks[:320000]
        elif subset == 'val':
            self.images = self.images[320000:325000]
            self.masks = self.masks[320000:325000]
        elif subset == 'test':
            self.images = self.images
            self.masks = self.masks

        print('{} set :'.format(subset), len(self.images), 'its')

        assert (len(self.images) == len(self.masks))
        for i in range(len(self.images)):
            image_index = self.images[i].split('/')[-1].split('.')[0]
            mask_index = self.masks[i].split('/')[-1].split('.')[0]
            assert (image_index == mask_index)
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index])
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
