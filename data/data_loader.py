import os
import sys
import caffe
import glob
import PIL
import cv2
import PIL.Image as Image
import plyvel
import random
import leveldb
import numpy as np
import torch
import torchvision

from caffe.proto import caffe_pb2
import torch.utils.data as data


def data_loader(dataset_file):
    db = leveldb.LevelDB(dataset_file)
    datum = caffe_pb2.Datum()

    img1s = []
    img2s = []
    labels = []

    for key, value in db.RangeIter():
        datum.ParseFromString(value)

        label = datum.label
        data = caffe.io.datum_to_array(datum)

        # split data from 6-channel image into 2 3-channel images
        img1 = data[:3, :, :]
        img2 = data[3:, :, :]

        labels.append(label)
        img1s.append(img1)
        img2s.append(img2)

    return labels, img1s, img2s


def binartproto2npy(mean_file):
    blob = caffe_pb2.BlobProto()
    blob.ParseFromString(open(mean_file, 'rb').read())
    arr = np.array(caffe.io.blobproto_to_array(blob))
    out = arr[0]
    return out


def get_mean_and_std(dataset):
    """
        Compute the mean and std value of dataset.
    """

    print('==> Computing mean and std..')
    mean = torch.zeros(3)
    std = torch.zeros(3)

    files = glob.glob(os.path.join(dataset, '*.png'))
    for img in files:
        im = np.array(Image.open(img).resize((128, 128), Image.BILINEAR), dtype=np.float64)
        im = im.transpose((2, 1, 0))
        for j in range(3):
            mean[j] += im[j, :, :].mean()
            std[j] += im[j, :, :].std()
    mean.div_(len(files) * 255)
    std.div_(len(files) * 255)

    return mean, std


class SiameseNetworkDataLoader(data.Dataset):
    def __init__(self, dataset_file, split='train', num_pair=256000, img_size=(128, 128), transform=None):
        self.split = split
        self.dataset_file = os.path.join(dataset_file, self.split)
        self.num_pair = num_pair
        self.img_size = img_size
        self.transform = transform
        self.num_classes = 2

    def __len__(self):
        return self.num_pair
        
    def __getitem__(self, index):
        files = glob.glob(os.path.join(self.dataset_file, '*.png'))

        img1 = random.choice(files)
        img1_name = img1.split('/')[-1].split('-')[0]

        # We need to make sure the number of positive and negative pair to be same 1:1
        get_same_class = random.randint(0, 1)
        if get_same_class:
            while True:
                # keep looping untill find the same class image
                img2 = random.choice(files)
                img2_name = img2.split('/')[-1].split('-')[0]
                if img1_name == img2_name:
                    break
        else:
            while True:
                img2 = random.choice(files)
                img2_name = img2.split('/')[-1].split('-')[0]
                if img1_name != img2_name:
                    break

        img1 = cv2.imread(img1)
        img1 = np.array(cv2.resize(img1, self.img_size), dtype=np.float64)
        img2 = cv2.imread(img2)
        img2 = np.array(cv2.resize(img2, self.img_size), dtype=np.float64)
        label = get_same_class

        assert img1.shape == img2.shape

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        else:
            img1 = np.transpose(img1, (2, 1, 0))
            img2 = np.transpose(img2, (2, 1, 0))

        label = torch.from_numpy(np.array([label], dtype=np.int64))  # convert label from int to LongTensor

        return label, img1, img2

    def collate_fn(self, batch):
        """Pad image pairs and targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of image pairs, labels.

        Returns:
          padded stacked targets, images_1(set), images_2(set).
        """

        labels = [x[0] for x in batch]
        images_1 = [x[1] for x in batch]
        images_2 = [x[2] for x in batch]

        batch_size = len(labels)
        inputs_1 = torch.zeros(batch_size, 3, self.img_size[1], self.img_size[0])
        inputs_2 = torch.zeros(batch_size, 3, self.img_size[1], self.img_size[0])

        targets = []

        for idx in range(batch_size):
            inputs_1[idx] = images_1[idx]
            inputs_2[idx] = images_2[idx]

            targets.append(labels[idx])

        return torch.stack(targets), inputs_1, inputs_2


if __name__ == "__main__":
    dataset_file = '/home/pingguo/ril-server/PycharmProject/database/faceDatasetCV/face_31&11/train'
    mean_file = '/home/pingguo/PycharmProject/InfantFaceVerification/data/mean31Kids_new_aligned.binaryproto'

    # dataset_folder = torchvision.datasets.DatasetFolder()
    # dataset = SiameseNetworkDataLoader(dataset_file=dataset_file, transform=None)

    # label, img1, img2 = data_loader(leveldb_file)
    # out = binartproto2npy(mean_file)
    # print(out)

    import time

    start_time = time.time()

    mean = torch.zeros(3)
    std = torch.zeros(3)
    files = glob.glob(os.path.join(dataset_file, '*.png'))
    for img in files:
        im = np.array(Image.open(img).resize((128, 128), Image.BILINEAR), dtype=np.float64)
        im = im.transpose((2, 1, 0))
        for j in range(3):
            mean[j] += im[j, :, :].mean()
            std[j] += im[j, :, :].std()
    mean.div_(len(files) * 255)
    std.div_(len(files) * 255)
    # mean, std = get_mean_and_std(dataset)
    print(mean, std)

    print("Inference time: {}s".format(time.time() - start_time))