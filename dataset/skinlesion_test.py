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
    test_image_list = []
    test_label_list = []

    for filename in data["traindata"]:
        img = cv2.imread(filename)
        train_image_list.append(img)
    for filename in data["trainlabel"]:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        train_label_list.append(img)
    for filename in data["testdata"]:
        img = cv2.imread(filename)
        test_image_list.append(img)
    for filename in data["testlabel"]:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (248, 248), interpolation=cv2.INTER_NEAREST)
        test_label_list.append(img)


    return train_image_list, train_label_list, test_image_list, test_label_list

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

    file = root + '/myTraining_Data248/'
    file_ground = root + '/myTraining_Label248/'
    file_valid = root + '/myValid_Data248/'
    file_ground_valid = root + '/myValid_Label248/'

    # image_list = np.zeros((300, 248,248,3), dtype='float32')
    # label_list = np.zeros((300, 248,248), dtype='uint8')
    # image_unlabel_list = np.zeros((1851, 248, 248, 3), dtype='float32')
    # label_unlabel_list = np.ones((1851, 248, 248), dtype='uint8')
    # label_unlabel_list = label_unlabel_list * 2

    image_list = []
    label_list = []
    image_unlabel_list = []
    label_unlabel_list = []

    random_valid = [61, 0,  88,  90, 111,  63,  79, 103,  71,  97,  35,  81,  97,  87,  43, 102,  16,  69, 136,  17,  33]
    file_box_valid = glob.glob(file_valid + '*.jpg')
    ground_box_valid = glob.glob(file_ground_valid + '*.png')
    file_box_valid.sort()
    ground_box_valid.sort()
    
    select_box_valid =  [file_box_valid[i] for i in random_valid]
    select_ground_valid = [ground_box_valid[i] for i in random_valid]


    #  279 in training file and 21 in validation file
    f = np.loadtxt('data_id/skin_id300_test.txt', dtype=int)
    index = f[0:297]

    file_box = glob.glob(file + '*.jpg')
    ground_box = glob.glob(file_ground + '*.png')
    file_box.sort()
    ground_box.sort()
    select_box =  [file_box[i] for i in index]
    select_ground = [ground_box[i] for i in index]

    select_box = [select_box[i] for i in range(len(select_box)) if not select_box[i] in select_box[:i]]
    select_ground = [select_ground[i] for i in range(len(select_ground)) if not select_ground[i] in select_ground[:i]]

    select_box_left = [i for i in file_box if i not in select_box]
    select_ground_left = [i for i in ground_box if i not in select_ground]
    select_box_left2 = [i for i in file_box_valid if i not in select_box_valid]
    select_ground_left2 = [i for i in ground_box_valid if i not in select_ground_valid]

    train_name = []
    for filename in select_box:
        img = cv2.imread(filename)
        train_name.append(filename)
        image_list.append(img)
    for filename in select_box_valid:
        train_name.append(filename)
        img = cv2.imread(filename)
        image_list.append(img)
    for filename in select_ground:
        label = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        label_list.append(label)
    for filename in select_ground_valid:
        label = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        label_list.append(label)


    #  load unlabelled data
    for filename in select_box_left:
        img = cv2.imread(filename)
        image_unlabel_list.append(img)
    for filename in select_box_left2:
        img = cv2.imread(filename)
        image_unlabel_list.append(img)
    for filename in select_ground_left:
        label = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        label_unlabel_list.append(label)
    for filename in select_ground_left2:
        label = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        label_unlabel_list.append(label)

    test_image = []
    test_label = []
    test_name = []
    test_list = glob.glob(root + "/myTest_Data248/*.jpg")
    label_test_list = glob.glob(root + "/myTest_Label512/*.png")
    label_test_list.sort()
    test_list.sort()

    for filename in test_list:
        test_name.append(filename)
        test_image.append(cv2.imread(filename))
    for filename in label_test_list:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (248, 248), interpolation=cv2.INTER_NEAREST)
        test_label.append(img)

    # path_train_data = glob.glob(root + 'myTraining_Data248/*.jpg')
    # path_valid_data = glob.glob(root + 'myValid_Data248/*.jpg')
    # path_test_data = glob.glob(root + 'myTest_Data248/*.jpg')

    # #  fix load files seq
    # path_train_data.sort()
    # path_valid_data.sort()
    # path_test_data.sort()

    ##  index of labeled data
    # index = list(range(0,len(path_train_data)))
    # np.random.shuffle(index)
    # train_labeled_idxs = index[:num_labels]
    # train_unlabeled_idxs = index[num_labels:]

    # #  index of fixed labeled data
    # if num_labels < 2000:
    #     a = np.loadtxt("data_id/skin_id"+str(num_labels)+".txt", dtype='str')
    #     a = [root + "myTraining_Data248/" + item for item in a]
    #     train_labeled_idxs = [path_train_data.index(item) for item in a]
    #     train_unlabeled_idxs = list(set(list(range(len(path_train_data)))) - set(train_labeled_idxs))
    # else:
    #     train_labeled_idxs = path_train_data
    #     train_unlabeled_idxs = []
    #
    # # label seq
    # path_train_label = ['/'.join(item.replace("myTraining_Data248", "myTraining_Label248").split("/")[:-1]) +"/"+
    #                     item.split("/")[-1][:-4]+"_segmentation.png" for item
    #                     in path_train_data]
    # path_valid_label = ['/'.join(item.replace("myValid_Data248", "myValid_Label248").split("/")[:-1]) +"/"+
    #                     item.split("/")[-1][:-4]+"_segmentation.png" for item
    #                     in path_valid_data]
    # path_test_label = ['/'.join(item.replace("myTest_Data248", "myTest_Label512").split("/")[:-1]) +"/"+
    #                     item.split("/")[-1][:-4]+"_segmentation.png" for item
    #                     in path_test_data]
    #
    # data = {"traindata": path_train_data,
    #         "trainlabel": path_train_label,
    #         "valdata": path_valid_data,
    #         "vallabel": path_valid_label,
    #         "testdata": path_test_data,
    #         "testlabel": path_test_label}
    #
    # # load data
    # train_data, train_label, test_data, test_label = get_skinlesion(data)
    #
    # val_name = path_valid_data
    # test_name= path_test_data
    # train_name = path_train_data


    train_labeled_dataset = skinlesion_labeled(image_list, label_list,name=train_name,
                                               transform=transform_train)
    train_unlabeled_dataset = skinlesion_unlabeled(image_unlabel_list, label_unlabel_list,
                                                   transform=TransformTwice(transform_train))

    test_dataset = skinlesion_labeled(test_image, test_label, name=test_name, transform=transform_val)

    print(f"#Labeled: {len(train_labeled_dataset)} #Unlabeled: {len(train_unlabeled_dataset)} #Val: {len(test_dataset)}")

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