import cv2
import numpy as np
import glob
import torch.utils.data as data
import numpy as np
from PIL import Image
import torchvision
import torch
from skimage.transform import resize

def get_skinlesion(data):
    print('-'*30)
    print('Loading images...')
    print('-'*30)

    train_image_list = []
    train_label_list = []
    val_image_list = []
    val_label_list = []
    test_image_list = []
    test_label_list = []

    for filename in data["traindata"]:
        img = cv2.imread(filename)
        train_image_list.append(img)
    for filename in data["trainlabel"]:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        train_label_list.append(img)
    for filename in data["valdata"]:
        img = cv2.imread(filename)
        val_image_list.append(img)
    for filename in data["vallabel"]:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        val_label_list.append(img)
    for filename in data["testdata"]:
        img = cv2.imread(filename)
        test_image_list.append(img)
    for filename in data["testlabel"]:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (248, 248), interpolation=cv2.INTER_NEAREST)
        test_label_list.append(img)


    return train_image_list, train_label_list, val_image_list, val_label_list, test_image_list, test_label_list

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp, target):
        out1, _ = self.transform(inp, target)
        # out2, _ = self.transform(inp, target)
        return out1, out1


class TransformRot:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp, target):
        out1, _ = self.transform(inp, target)
        # out2, _ = self.transform(inp, target)
        return out1, out1


def get_skinlesion_dataset(root, num_labels, transform_train=None, transform_val=None, transform_forsemi=None):

    file = root + '/dataset/resized_dataset/'
    random_idx = np.genfromtxt(root + '/dataset/random_idx.txt', dtype="str")

    labeled_idx = random_idx[:num_labels]
    image_list = []
    label_list = []
    train_name = []
    for filename in labeled_idx:
        train_name.append(filename)
        img = cv2.imread(file + filename)
        image_list.append(img)
    for filename in labeled_idx:
        label = cv2.imread(file + filename[:-4] + '.bmp', cv2.IMREAD_GRAYSCALE)
        label_list.append(label)


    # unlabeled_idx
    unlabel_idx = random_idx[num_labels:-num_labels]
    image_unlabel_list = []
    for filename in unlabel_idx:
        img = cv2.imread(file+filename)
        image_unlabel_list.append(img)

    # test_data
    test_idx = random_idx[-num_labels:]
    print (test_idx)
    exit(0)
    image_test_list = []
    name_test_list = []
    label_test_list = []
    for filename in test_idx:
        img = cv2.imread(file+filename)
        name_test_list.append(filename)
        image_test_list.append(img)
    for filename in test_idx:
        label = cv2.imread(file + filename[:-4] + '.bmp', cv2.IMREAD_GRAYSCALE)
        label_test_list.append(label)

    train_labeled_dataset = skinlesion_labeled(image_list, label_list, name=train_name, transform=transform_train)
    train_unlabeled_dataset = skinlesion_unlabeled(image_unlabel_list, image_unlabel_list,
                                                   transform=TransformTwice(transform_train))

    test_dataset = skinlesion_labeled(image_test_list, label_test_list, name=name_test_list, transform=transform_val)

    print(
        f"#Labeled: {len(train_labeled_dataset)} #Unlabeled: {len(train_unlabeled_dataset)} #Val: {len(test_dataset)}")

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255




def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')


class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x


class RandomFlip(object):
    """Flip randomly the image.
    """

    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()


class GaussianNoise(object):
    """Add gaussian noise to the image.
    """

    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x


class ToTensor(object):
    """Transform the image to tensor.
    """

    def __call__(self, x):
        x = torch.from_numpy(x)
        return x



class skinlesion_labeled(data.Dataset):

    def __init__(self, data, label, name = None,
                 transform=None):

        self.data = data
        self.targets = label
        self.transform = transform
        self.name = name



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img, target = self.transform(img, target)

        if self.name is not None:
            return img, target, self.name[index]
        else:
            return img, target



    def __len__(self):
        return len(self.data)



class skinlesion_unlabeled(data.Dataset):

    def __init__(self, data, label,
                 transform=None):

        self.data = data
        self.targets = [-1*np.ones_like(label[item]) for item in range(0,len(label))]

        self.transform = transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.data)
