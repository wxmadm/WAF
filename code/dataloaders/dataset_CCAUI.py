import itertools
import os
import random
import re
from glob import glob
import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from skimage import exposure
import torchvision.transforms.functional as TF
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Dataset
import copy
from pytorch_wavelets import DWTForward, DWTInverse
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, labeled_type="labeled", labeled_ratio=10, split='train', transform=None, fold=1, cross_val=True):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.labeled_type = labeled_type
        #self.all_volumes = sorted(os.listdir(self._base_dir + "/all_volumes"))
        train_ids = sorted(os.listdir(self._base_dir + "/train"))
        val_ids = sorted(os.listdir(self._base_dir + "/val"))
        train_files = set(os.listdir("/home/wxm/code/SSLM4-LIS/data/CCAUI/train"))
        val_files = set(os.listdir("/home/wxm/code/SSLM4-LIS/data/CCAUI/val"))
        print("Train-Val Overlap:", train_files.intersection(val_files))
        if self.split == 'train':
            self.all_slices = os.listdir("/home/wxm/code/SSLM4-LIS/data/CCAUI/train_val")#去加载 他的切片，这个切片数据包含很多 ，所有train的 切片，有标签和无标签都包含进去
            self.sample_list = []
            labeled_ids = sorted(os.listdir(self._base_dir + "200/labeled"))
            unlabeled_ids = sorted(os.listdir(self._base_dir + "200/unlabeled"))
            if self.labeled_type == "labeled":
                print("Labeled patients IDs", labeled_ids)
                for ids in labeled_ids:
                    prefix = ids.replace(".h5", "")
                    pattern = re.compile(r'^{}(\.|_|$)'.format(re.escape(prefix)))  # 严格匹配前缀，后面紧跟 '.' 或 '_' 或 结尾
                    new_data_list = list(filter(lambda x: pattern.match(x), self.all_slices))
                    self.sample_list.extend(new_data_list)  #这个操作跟切片有关系 不需要太详细的了解
                print("total labeled {} samples".format(len(self.sample_list)))#生成所有的切片数据到底有多少
                print("ok stop")
            else:
                print("Unlabeled patients IDs", unlabeled_ids)
                for ids in unlabeled_ids:
                    prefix = ids.replace(".h5", "")
                    pattern = re.compile(r'^{}(\.|_|$)'.format(re.escape(prefix)))
                    new_data_list = list(filter(lambda x: pattern.match(x), self.all_slices))
                    self.sample_list.extend(new_data_list)
                print("total unlabeled {} samples".format(len(self.sample_list)))
        elif self.split == 'val':
            # 加载所有的切片数据
            self.all_slices = os.listdir("/home/wxm/code/SSLM4-LIS/data/CCAUI/train_val")
            print("val_ids", val_ids)

            self.sample_list = []

            # 遍历 val_ids，匹配 self.all_slices 里的相关切片
            for ids in val_ids:
                prefix = ids.replace(".h5", "")
                pattern = re.compile(r'^{}(\.|_|$)'.format(re.escape(prefix)))
                new_data_list = list(filter(lambda x: pattern.match(x), self.all_slices))
                self.sample_list.extend(new_data_list)

            print("Total validation samples:", len(self.sample_list))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/train/{}".format(case), 'r') #切换数据集，进行数据增强 作用
        else:
            h5f = h5py.File(self._base_dir +
                            "/val/{}".format(case), 'r')
        if self.split == "train":
            image = h5f['image'][:]
            label = h5f["label"][:]
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)#读取 图像和标签 ，应用变换，返回增强后的样本
        else:
            image = h5f['image'][:]
            label = h5f['label'][:].astype(np.int16)
            sample = {'image': image, 'label': label}
        sample["idx"] = case.split("_")[0]
        return sample


def random_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    return image, label


def random_noise(image, label, mu=0, sigma=0.1):
    noise = np.clip(sigma * np.random.randn(image.shape[0], image.shape[1]),
                    -2 * sigma, 2 * sigma)
    noise = noise + mu
    image = image + noise
    return image, label


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * (t**(n-i)) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array(
        [bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation(x, label, prob=0.5):
    if random.random() >= prob:
        return x, label
    points = [[0, 0], [random.random(), random.random()], [
        random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x, label


def random_rescale_intensity(image, label):
    image = exposure.rescale_intensity(image)
    return image, label


def random_equalize_hist(image, label):
    image = exposure.equalize_hist(image)
    return image, label



def random_equalize_hist(image, label):
    image = exposure.equalize_hist(image)
    return image, label


class RandomGenerator_Strong_Weak(object):
    def __init__(self, output_size=(128, 128)):
        self.output_size = output_size
        self.dwt = DWTForward(J=1, wave='db3', mode='zero')

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 缩放到目标尺寸
        x, y = image.shape
        image = zoom(image, (self.output_size[0]/x, self.output_size[1]/y), order=0)
        label = zoom(label, (self.output_size[0]/x, self.output_size[1]/y), order=0)

        # === 弱增强路径 ===
        if random.random() > 0.5:
            image,  label = random_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label, cval=0)
        if random.random() > 0.5:
            image, label = random_noise(image, label)
        
        image_w = image.copy()


        # 转换为张量并小波变换
        image_w = torch.from_numpy(image_w.astype(np.float32))
        if image_w.dim() == 2:  # [H, W]
            image_w = image_w.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        # image_w_tensor = torch.from_numpy(image_w.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        Yl_w, Yh_w = self.dwt(image_w)


        # if random.random() > 0.5:
        #     Yh_w = normalize_high_freq(Yh_w)
        
        # 拼接小波分量（完全保持原格式）
        wavelet_input_w = torch.cat([
            Yl_w,
            Yh_w[0][:, :, 0:1].squeeze(2),
            Yh_w[0][:, :, 1:2].squeeze(2),
            Yh_w[0][:, :, 2:3].squeeze(2)
        ], dim=1)
        wavelet_input_w = wavelet_input_w[:, :, :128, :128]  # 确保尺寸一致
        image_w = wavelet_input_w.squeeze(0)  # [4, 128, 128]

        # === 强增强路径（在弱增强基础上）===
        if random.random() > 0.33:
            image, label = nonlinear_transformation(image, label)
        elif random.random() < 0.66 and random.random() > 0.33:
            image, label = random_rescale_intensity(image, label)
        else:
            image, label = random_equalize_hist(image, label)


        image_s = image.copy()  # 基于弱增强结果
        # 转换为张量并小波变换
        image_s = torch.from_numpy(image_s.astype(np.float32))
        if image_s.dim() == 2:  # [H, W]
            image_s = image_s.unsqueeze(0).unsqueeze(0)
        Yl_s, Yh_s = self.dwt(image_s)

        # if random.random() > 0.5:
        #     Yh_s = normalize_high_freq(Yh_s)

        # 拼接小波分量（完全保持原格式）
        wavelet_input_s = torch.cat([
            Yl_s,
            Yh_s[0][:, :,0:1].squeeze(2),
            Yh_s[0][:, :,1:2].squeeze(2),
            Yh_s[0][:, :,2:3].squeeze(2)
        ], dim=1)
        wavelet_input_s = wavelet_input_s[:, :, :128, :128]  # 确保尺寸一致
        image_s = wavelet_input_s.squeeze(0)
        # 标签转换（保持原格式）
        label = torch.from_numpy(label.astype(np.int16))

        # 返回结果（完全保持原格式）

        sample = {'image_w': image_w, 'image_s': image_s, 'label': label}
        return sample

    
def normalize_high_freq(Yh):
    Yh_norm = []
    for band in Yh[0]:  # band: [C, H, W]
        mean = band.mean(dim=[1, 2], keepdim=True)  # 对每个通道的空间平均
        std = band.std(dim=[1, 2], keepdim=True)
        norm_band = (band - mean) / (std + 1e-5)
        Yh_norm.append(norm_band)
    return [torch.stack(Yh_norm)]


# class RandomGenerator_Strong_Weak(object):
#     def __init__(self, output_size):
#         self.output_size = output_size
#         self.dwt = DWTForward(J=1, wave='db3', mode='zero')

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         # Handle tensor or NumPy input
#         if torch.is_tensor(image):
#             image = image.squeeze().numpy()  # Convert to NumPy, remove batch/channel dims
#         if torch.is_tensor(label):
#             label = label.squeeze().numpy()

#         x, y = image.shape  # Now [256, 256]
#         image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
#         label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
#         image = torch.from_numpy(image.astype(np.float32))
#         if image.dim() == 2:
#             image = image.unsqueeze(0).unsqueeze(0)  # [1, 1, 256, 256]
#         Yl, Yh = self.dwt(image)
#         Yh_w = [yh.clone() for yh in Yh]
#         if random.random() > 0.5:
#             for i in range(Yh_w[0].shape[2]):
#                 Yh_w[0][:, :, i] += torch.randn_like(Yh_w[0][:, :, i]) * 0.05
#         y_3 = Yh_w[0][:, :, 2:3].squeeze(2)
#         wavelet_input_w = torch.cat([
#             Yl,
#             Yh_w[0][:, :, 0:1].squeeze(2),
#             Yh_w[0][:, :, 1:2].squeeze(2),
#             Yh_w[0][:, :, 2:3].squeeze(2)
#         ], dim=1)[:, :, :128, :128]
#         image_w = wavelet_input_w.squeeze(0)
#         Yh_s = [yh.clone() for yh in Yh]
#         if random.random() > 0.5:
#             edge_mask = self._edge_mask(Yh_s[0], threshold=0.1)
#             for i in range(Yh_s[0].shape[2]):
#                 #Yh_s[0][:, :, i] *= random.uniform(0.5, 1.5)
#                 Yh_s[0][:, :, i] *= edge_mask
#                 Yh_s[0][:, :, i] += torch.randn_like(Yh_s[0][:, :, i]) * 0.5
#         if random.random() > 0.5:
#             mask = torch.rand(Yh_s[0].shape) > 0.5
#             Yh_s[0] = Yh_s[0] * mask.float()
#         if random.random() > 0.5:
#             angle = random.uniform(-10, 10)
#             Yl_rot = torch.rot90(Yl, k=int(angle // 90), dims=[2, 3]) if abs(angle) >= 5 else Yl
#             Yl_s = Yl_rot
#         else:
#             Yl_s = Yl
#         wavelet_input_s = torch.cat([
#             Yl_s,
#             Yh_s[0][:, :, 0:1].squeeze(2),
#             Yh_s[0][:, :, 1:2].squeeze(2),
#             y_3

#         ], dim=1)[:, :, :128, :128]
#         image_s = wavelet_input_s.squeeze(0)
#         label = torch.from_numpy(label.astype(np.int16))
#         sample = {'image_w': image_w, 'image_s': image_s, 'label': label}
#         return sample

#     def _edge_mask(self, high_freq, threshold=0.1):
#         # 边缘检测：基于高频分量的梯度
#         grad = torch.abs(high_freq).mean(dim=2)  # [1, 1, 130, 130]
#         grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
#         mask = (grad > threshold).float()  # [1, 1, 130, 130]
#         return mask










class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label, cval=0)
        if random.random() > 0.5:
            image, label = random_noise(image, label)
        if random.random() > 0.33:
            image, label = nonlinear_transformation(image, label)
        elif random.random() < 0.66 and random.random() > 0.33:
            image, label = random_rescale_intensity(image, label)
        elif random.random() > 0.66:
            image, label = random_equalize_hist(image, label)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int16))
        sample = {'image': image, 'label': label}
        return sample
