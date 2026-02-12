
import argparse
import os
import shutil
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
from pytorch_wavelets import DWTForward
from PIL import Image
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/wxm/code/SSLM4-LIS/data/DDTI/test',
                    help='Path to test data folder')
parser.add_argument('--exp', type=str,
                    default='DDTI/Fix-match-oriwavlet-Yls/5', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data (not used in testing)')
parser.add_argument('--pred_save_path', type=str, default='/home/wxm/code/SSLM4-LIS/result/DDTI/Fix-match-dwt-SE-SA/150', help='Path to save predicted labels as PNG')
parser.add_argument('--save_mode_path', type=str, default='/home/wxm/code/model/150/DDTI/Fix-match-dwt-SE-SA/unet/unet_best_model.pth', help='Path to save predicted labels as PNG')
 

def calculate_metric_percase(pred, gt, spacing=(1.0, 1.0)):
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match GT shape {gt.shape}")
    if pred.sum() > 0 and gt.sum() > 0:
        pred[pred > 0] = 1
        gt[gt > 0] = 1
        dice = metric.binary.dc(pred, gt)
        asd = metric.binary.asd(pred, gt, voxelspacing=spacing)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
    else:
        dice = 0.0
        hd95 = 100.0
        asd = 20.0
    return dice, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(os.path.join(FLAGS.root_path, f"{case}.h5"), 'r')
    image = h5f['image'][:]  # [256, 256]
    label = h5f['label'][:]  # [256, 256]

    # 目标尺寸为 [256, 256]
    target_size = (256, 256)

    # 小波变换
    dwt = DWTForward(J=1, wave='db3', mode='zero')
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()  # [1, 1, 256, 256]
    Yl, Yh = dwt(image_tensor)
    Yl = Yl[:, :, :128, :128]  # [1, 1, 128, 128]
    Yh0 = Yh[0][:, :, :, :128, :128]  # [1, 1, 3, 128, 128]

    # 拼接子带
    input = torch.cat([
        Yl,
        Yh0[:, :, 0:1].squeeze(2),
        Yh0[:, :, 1:2].squeeze(2),
        Yh0[:, :, 2:3].squeeze(2)
    ], dim=1).cuda()  # [1, 4, 128, 128]

    net.eval()
    with torch.no_grad():
        if FLAGS.model == "unet_urds":
            out_main, _, _, _ = net(input)
        else:
            out_main = net(input)
        out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)  # [128, 128] 或 [256, 256]
        out = out.cpu().detach().numpy()

        # 根据模型输出调整到目标尺寸
        if out.shape != target_size:
            prediction = zoom(out, (target_size[0] / out.shape[0], target_size[1] / out.shape[1]), order=0)
        else:
            prediction = out

    # 调试信息
    print(f"Case: {case}")
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction unique values: {np.unique(prediction)}")
    print(f"Label unique values: {np.unique(label)}")
    print(f"Class 1 in prediction: {(prediction == 1).sum()} pixels")
    print(f"Class 1 in label: {(label == 1).sum()} pixels")
    print(f"Class 2 in prediction: {(prediction == 2).sum()} pixels")
    print(f"Class 2 in label: {(label == 2).sum()} pixels")

    # 计算指标
    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    pred_save_path = FLAGS.pred_save_path
    # 保存结果
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, os.path.join(test_save_path, f"{case}_pred.nii.gz"))
    sitk.WriteImage(img_itk, os.path.join(test_save_path, f"{case}_img.nii.gz"))
    sitk.WriteImage(lab_itk, os.path.join(test_save_path, f"{case}_gt.nii.gz"))
    pred_img = Image.fromarray(prediction.astype(np.uint8))
    pred_img.save(os.path.join(pred_save_path, f"{case}_pred.png"))

    return first_metric, second_metric


import csv

def Inference(FLAGS):
    test_folder = FLAGS.root_path
    image_list = sorted([f.split(".h5")[0] for f in os.listdir(test_folder) if f.endswith(".h5")])

    test_save_path = os.path.join("model/", FLAGS.exp, "prediction")
    pred_save_path = FLAGS.pred_save_path

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    if os.path.exists(pred_save_path):
        shutil.rmtree(pred_save_path)
    os.makedirs(pred_save_path)

    net = net_factory(net_type=FLAGS.model, in_chns=4, class_num=FLAGS.num_classes)
    save_mode_path = FLAGS.save_mode_path
    checkpoint = torch.load(save_mode_path)
    net.load_state_dict(checkpoint)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    dice_class1_list = []
    dice_class2_list = []
    first_total = 0.0
    second_total = 0.0

    # CSV 写入
    csv_path = os.path.join(pred_save_path, "dice_scores.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Case", "Dice_Class1", "Dice_Class2"])

        for case in tqdm(image_list):
            first_metric, second_metric = test_single_volume(case, net, test_save_path, FLAGS)
            first_total += np.asarray(first_metric)
            second_total += np.asarray(second_metric)

            dice_class1 = first_metric[0]
            dice_class2 = second_metric[0]
            dice_class1_list.append(dice_class1)
            dice_class2_list.append(dice_class2)

            csv_writer.writerow([case, f"{dice_class1:.4f}", f"{dice_class2:.4f}"])

        # 写入统计摘要
        csv_writer.writerow([])
        csv_writer.writerow(["Summary Statistics"])

        for idx, dice_list in enumerate([dice_class1_list, dice_class2_list], start=1):
            dice_array = np.array(dice_list)
            mean_val = np.mean(dice_array)
            var_val = np.var(dice_array)
            std_val = np.std(dice_array)
            csv_writer.writerow([
                f"Dice_class{idx} Mean", f"{mean_val:.4f}",
                "Variance", f"{var_val:.4f}",
                "StdDev", f"{std_val:.4f}"
            ])

    avg_metric = [first_total / len(image_list), second_total / len(image_list)]
    return avg_metric



if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)

    print("Class 1 metrics (Dice, HD95, ASD):", [f"{v:.4f}" for v in metric[0]])
    print("Class 2 metrics (Dice, HD95, ASD):", [f"{v:.4f}" for v in metric[1]])

    avg_dice = (metric[0][0] + metric[1][0]) / 2
    avg_hd95 = (metric[0][1] + metric[1][1]) / 2
    avg_asd = (metric[0][2] + metric[1][2]) / 2

    print("Average Dice:", f"{avg_dice:.4f}")
    print("Average HD95:", f"{avg_hd95:.4f}")
    print("Average ASD:", f"{avg_asd:.4f}")