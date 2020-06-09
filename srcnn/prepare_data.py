## Major code belongs to: MarkPrecursor @github

import os
import cv2
import h5py
import numpy

class DataProcessor:
    DATA_PATH = "./Train/"
    TEST_PATH = "./Test/"

    # BORDER_CUT = 8
    BLOCK_STEP = 16
    BLOCK_SIZE = 32

    Random_Crop = 30
    Patch_size = 32
    label_size = 20
    conv_side = 6
    scale = 2

    def change_path(self, data_path="./Train/", test_path="./Test/"):
        self.DATA_PATH = data_path
        self.TEST_PATH = test_path

    def change_factors(self, scale=2, conv_side=6, label_size=20, patch_size=32, random_crop=30, block_step=16, block_size=32):
        self.scale = scale
        self.conv_side = conv_side
        self.label_size = label_size
        self.Patch_size = patch_size
        self.Random_Crop = random_crop
        self.BLOCK_STEP = block_step
        self.BLOCK_SIZE = block_size

    def prepare_data(self, _path):
        names = os.listdir(_path)
        names = sorted(names)
        nums = names.__len__()

        data = numpy.zeros((nums * self.Random_Crop, 1, self.Patch_size, self.Patch_size), dtype=numpy.double)
        label = numpy.zeros((nums * self.Random_Crop, 1, self.label_size, self.label_size), dtype=numpy.double)

        for i in range(nums):
            name = _path + names[i]
            hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
            shape = hr_img.shape

            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
            hr_img = hr_img[:, :, 0]

            # two resize operation to produce training data and labels
            lr_img = cv2.resize(hr_img, (shape[1] // self.scale, shape[0] // self.scale))
            lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

            # produce self.Random_Crop random coordinate to crop training img
            Points_x = numpy.random.randint(0, min(shape[0], shape[1]) - self.Patch_size, self.Random_Crop)
            Points_y = numpy.random.randint(0, min(shape[0], shape[1]) - self.Patch_size, self.Random_Crop)

            for j in range(self.Random_Crop):
                lr_patch = lr_img[Points_x[j]: Points_x[j] + self.Patch_size, Points_y[j]: Points_y[j] + self.Patch_size]
                hr_patch = hr_img[Points_x[j]: Points_x[j] + self.Patch_size, Points_y[j]: Points_y[j] + self.Patch_size]

                lr_patch = lr_patch.astype(float) / 255.
                hr_patch = hr_patch.astype(float) / 255.

                data[i * self.Random_Crop + j, 0, :, :] = lr_patch
                if self.conv_side != 0:
                    label[i * self.Random_Crop + j, 0, :, :] = hr_patch[self.conv_side: -self.conv_side, self.conv_side: -self.conv_side]
                else:
                    label[i * self.Random_Crop + j, 0, :, :] = hr_patch
                # cv2.imshow("lr", lr_patch)
                # cv2.imshow("hr", hr_patch)
                # cv2.waitKey(0)
        return data, label


    def prepare_crop_data(self, _path):
        names = os.listdir(_path)
        names = sorted(names)
        nums = names.__len__()

        data = []
        label = []

        for i in range(nums):
            name = _path + names[i]
            hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
            hr_img = hr_img[:, :, 0]
            shape = hr_img.shape

            # two resize operation to produce training data and labels
            lr_img = cv2.resize(hr_img, (shape[1] // self.scale, shape[0] // self.scale))
            lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

            width_num = (shape[0] - (self.BLOCK_SIZE - self.BLOCK_STEP) * 2) // self.BLOCK_STEP
            height_num = (shape[1] - (self.BLOCK_SIZE - self.BLOCK_STEP) * 2) // self.BLOCK_STEP
            for k in range(width_num):
                for j in range(height_num):
                    x = k * self.BLOCK_STEP
                    y = j * self.BLOCK_STEP
                    hr_patch = hr_img[x: x + self.BLOCK_SIZE, y: y + self.BLOCK_SIZE]
                    lr_patch = lr_img[x: x + self.BLOCK_SIZE, y: y + self.BLOCK_SIZE]

                    lr_patch = lr_patch.astype(float) / 255.
                    hr_patch = hr_patch.astype(float) / 255.

                    lr = numpy.zeros((1, self.Patch_size, self.Patch_size), dtype=numpy.double)
                    hr = numpy.zeros((1, self.label_size, self.label_size), dtype=numpy.double)

                    lr[0, :, :] = lr_patch
                    if self.conv_side != 0:
                        hr[0, :, :] = hr_patch[self.conv_side: -self.conv_side, self.conv_side: -self.conv_side]
                    else:
                        hr[0, :, :] = hr_patch

                    data.append(lr)
                    label.append(hr)

        data = numpy.array(data, dtype=float)
        label = numpy.array(label, dtype=float)
        return data, label

    def generate_data(self):
        data, label = self.prepare_crop_data(self.DATA_PATH)
        print("Writing train data ...")
        write_hdf5(data, label, "train.h5")
        data, label = self.prepare_data(self.TEST_PATH)
        print("Writing test data ...")
        write_hdf5(data, label, "test.h5")


def write_hdf5(data, labels, output_filename):
    """
    This function is used to save image data and its label(s) to hdf5 file.
    output_file.h5,contain data and label
    """

    x = data.astype(numpy.float32)
    y = labels.astype(numpy.float32)

    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)
        h.create_dataset('label', data=y, shape=y.shape)
        # h.create_dataset()


def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data = numpy.array(hf.get('data'))
        label = numpy.array(hf.get('label'))
        train_data = numpy.transpose(data, (0, 2, 3, 1))
        train_label = numpy.transpose(label, (0, 2, 3, 1))
        return train_data, train_label
