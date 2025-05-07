from PIL import Image
import numpy as np
import os
from glob import glob
import argparse
import scipy.spatial
import tqdm
import scipy.ndimage as ndimage
import scipy
import h5py


def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    distances, _ = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.0
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
            if sigma > 15:
                sigma = 15
        else:
            sigma = np.average(np.array(gt.shape)) / 2.0 / 2.0
        density += ndimage.filters.gaussian_filter(pt2d, sigma, mode="constant")
    return density


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
            if im_h < min_size:
                ratio = 1.0 * min_size / im_h
                im_h = min_size
                im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
            if im_w < min_size:
                ratio = 1.0 * min_size / im_w
                im_w = min_size
                im_h = round(im_h * ratio)
        else:
            ratio = 1.0

    return im_h, im_w, ratio


def generate_data(im_path):
    im = Image.open(im_path)
    im_w, im_h = im.size
    mat_path = im_path.replace("images", "gt").replace(".jpg", ".txt")
    points = []
    with open(mat_path, "r") as f:
        while True:
            point = f.readline()
            if not point:
                break
            point = point.split(" ")[:-1]
            points.append([float(point[0]), float(point[1])])
    points = np.array(points)
    if len(points) > 0:
        idx_mask = (
            (points[:, 0] >= 0)
            * (points[:, 0] <= im_w)
            * (points[:, 1] >= 0)
            * (points[:, 1] <= im_h)
        )
        points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    if rr != 1.0:
        im = im.resize((im_w, im_h), Image.LANCZOS)
        points = points * rr
    return im, points


def parse_args():
    parser = argparse.ArgumentParser(description="Data processing")
    parser.add_argument(
        "--origin-dir", default="jhu_crowd_v2.0", help="orginal data directory"
    )
    parser.add_argument(
        "--data-dir", default="./datasets/jhu", help="processed data directory"
    )
    parser.add_argument(
        "--min-size", type=int, default=512, help="minimum size for images"
    )
    parser.add_argument(
        "--max-size", type=int, default=1920, help="maximum size for images"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    save_dir = args.data_dir
    min_size = args.min_size
    max_size = args.max_size

    for phase in ["train", "val", "test"]:
        sub_dir = os.path.join(args.data_dir, phase)
        sub_dir_img = os.path.join(sub_dir, "images")
        sub_dir_pts = os.path.join(sub_dir, "gt_points")
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
            os.makedirs(sub_dir_img)
            os.makedirs(sub_dir_pts)
        im_list = sorted(
            glob(os.path.join(args.origin_dir, "{}/images".format(phase), "*.jpg"))
        )
        if phase == "train":
            sub_dir_den = os.path.join(sub_dir, "gt_den")
            if not os.path.exists(sub_dir_den):
                os.makedirs(sub_dir_den)
        for im_path in tqdm.tqdm(im_list):
            im, points = generate_data(im_path)
            name = os.path.basename(im_path)
            if phase == "train":
                w, h = im.size
                d = np.zeros((h, w))
                for j in range(len(points)):
                    point_x, point_y = points[j][0:2].astype("int")
                    if point_y >= h or point_x >= w:
                        continue
                    d[point_y, point_x] = 1
                d = gaussian_filter_density(d)
                with h5py.File(
                    os.path.join(sub_dir_den, "{}.h5".format(name.replace(".jpg", ""))),
                    "w",
                ) as hf:
                    hf["density_map"] = d
                print(
                    name, "GT_num:", len(points), "Density_sum: {:.2f}".format(d.sum())
                )
            im_save_path = os.path.join(sub_dir_img, name)
            im.save(im_save_path)
            gd_save_path = im_save_path.replace("jpg", "npy").replace(
                "images", "gt_points"
            )
            np.save(gd_save_path, points)
