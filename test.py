import argparse
from datetime import datetime
from glob import glob
import math
from matplotlib import pyplot as plt
import numpy as np
from datasets.dataset import Crowd
import torch
import os
from torch.utils.data import DataLoader
from model.model import mamba
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--data-dir", default="", help="training data directory")
    parser.add_argument("--save-dir", default="", help="model directory")
    parser.add_argument("--device", default="0", help="assign device")
    parser.add_argument(
        "--crop-size", type=int, default=2048, help="the crop size of the test image"
    )
    args = parser.parse_args()
    return args


def tensorToImage(img_tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
    )
    img_tensor = img_tensor.squeeze(0)
    img_tensor = inv_normalize(img_tensor)
    img_tensor = img_tensor.clamp(0, 1)
    np_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    np_img = np.clip(np_img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)

def visualize_density(density_tensor, file_name):
    # upsample density map 8x
    density_tensor = F.interpolate(
        density_tensor, scale_factor=8, mode="bilinear", align_corners=False
    )
    density_tensor = density_tensor.squeeze(0).squeeze(0)
    density_tensor = density_tensor.cpu().numpy()
    density_tensor /= np.max(density_tensor)
    plt.imsave("visual/" + file_name + "_density.png", density_tensor, cmap="jet")


if __name__ == "__main__":
    args = parse_args()
    # torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.strip()
    dataset = Crowd(args.data_dir, 512, 8, method="val")
    dataloader = DataLoader(dataset, 1, shuffle=False, pin_memory=False)
    model = mamba(25)
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    log_list = []
    model_list = sorted(glob(os.path.join(args.save_dir, "*.pth")))

    for model_path in model_list:
        epoch_minus = []
        model.load_state_dict(torch.load(model_path, device, weights_only=False))
        for inputs, count, name in dataloader:
            inputs = inputs.to(device)
            b, c, h, w = inputs.shape
            h, w = int(h), int(w)
            assert b == 1
            input_list = []
            c_size = args.crop_size
            if h >= c_size or w >= c_size:
                h_stride = int(math.ceil(1.0 * h / c_size))
                w_stride = int(math.ceil(1.0 * w / c_size))
                h_step = h // h_stride
                w_step = w // w_stride
                for i in range(h_stride):
                    for j in range(w_stride):
                        h_start = i * h_step
                        if i != h_stride - 1:
                            h_end = (i + 1) * h_step
                        else:
                            h_end = h
                        w_start = j * w_step
                        if j != w_stride - 1:
                            w_end = (j + 1) * w_step
                        else:
                            w_end = w
                        input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
                with torch.set_grad_enabled(False):
                    pre_count = 0.0
                    for idx, input in enumerate(input_list):
                        output = model(input)[0]
                        pre_count += torch.sum(output)
                res = count[0].item() - pre_count.item()
                epoch_minus.append(res)
            else:
                with torch.set_grad_enabled(False):
                    outputs, cls_score = model(inputs)
                    if args.visual_img:
                        mask = torch.argmax(cls_score, dim=1, keepdim=True) == 0
                        mask = F.interpolate(
                            mask.float(),
                            size=(inputs.size(2), inputs.size(3)),
                            mode="nearest",
                        )
                        mask = mask.squeeze(0).squeeze(0).cpu().numpy() * 255
                        mask = Image.fromarray(mask.astype(np.uint8), mode="L")
                        mask.save("visual/" + name[0] + "_mask.jpg")
                    res = count[0].item() - torch.sum(outputs).item()
                    epoch_minus.append(res)


        epoch_minus = np.array(epoch_minus)
        mse = np.sqrt(np.mean(np.square(epoch_minus)))
        mae = np.mean(np.abs(epoch_minus))
        log_str = "model_name {}, mae {}, mse {}".format(
            os.path.basename(model_path), mae, mse
        )
        log_list.append(log_str)
        print(log_str)

    date_str = datetime.strftime(datetime.now(), "%m%d-%H%M%S")
    with open(
        os.path.join(args.save_dir, "test_results_{}.txt".format(date_str)), "w"
    ) as f:
        for log_str in log_list:
            f.write(log_str + "\n")
        f.write("crop size: {}".format(args.crop_size))
