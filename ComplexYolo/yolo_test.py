import numpy as np
import torch
import cv2

import ComplexYolo.config.kitti_config as cnf

from data_process import kitti_data_utils, kitti_bev_utils
from ComplexYolo.utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format
from utils.evaluation_utils import rescale_boxes, post_processing_v2
from ComplexYolo.models.darknet2pytorch import Darknet

import open3d as o3d

def get_lidar(filename):
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

def preprocess_lidar(point_cloud):
    point_cloud = kitti_bev_utils.removePoints(point_cloud, cnf.boundary)
    rgb_map = kitti_bev_utils.makeBVFeature(point_cloud, cnf.DISCRETIZATION, cnf.boundary)

    # convert to tensor and add batch dimension
    rgb_map = torch.from_numpy(rgb_map).unsqueeze(0).float().to(device)

    return rgb_map


class Config:
    def __init__(self, pretrained_path, config_file):
        self.pretrained_path = pretrained_path
        self.cfgfile = config_file

        self.img_size = 608
        self.conf_thresh = 0.5
        self.nms_thresh = 0.5

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    configs = Config('../Complex-YOLOv4-Pytorch/checkpoints/complex_yolov4/complex_yolov4_mse_loss.pth',
                     '../Complex-YOLOv4-Pytorch/src/config/cfg/complex_yolov4.cfg')

    model = Darknet(cfgfile=configs.cfgfile, use_giou_loss=False)

    model.load_state_dict(torch.load(configs.pretrained_path, map_location=device))

    model.to(device)

    model.eval()

    point_cloud = get_lidar('../Complex-YOLOv4-Pytorch/dataset/kitti/testing/velodyne/000000.bin')

    rgb_map = preprocess_lidar(point_cloud)

    with torch.no_grad():
        outputs = model(rgb_map)

        detections = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)

        img_detections = []  # Stores detections for each image index
        img_detections.extend(detections)

        img_bev = rgb_map.squeeze().detach().cpu() * 255

        img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
        img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))
        for detections in img_detections:
            if detections is None:
                continue
            # Rescale boxes to original image
            detections = rescale_boxes(detections, configs.img_size, img_bev.shape[:2])
            for x, y, w, l, im, re, *_, cls_pred in detections:
                yaw = np.arctan2(im, re)
                # Draw rotated box
                kitti_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])



        img_path = '../Complex-YOLOv4-Pytorch/dataset/kitti/testing/image_2/000000.png'
        img_rgb = cv2.imread(img_path)
        calib = kitti_data_utils.Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))
        objects_pred = predictions_to_kitti_format(img_detections, calib, img_rgb.shape, configs.img_size)
        img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, False)

        img_bev = cv2.flip(cv2.flip(img_bev, 0), 1)

        out_img = merge_rgb_to_bev(img_rgb, img_bev, output_width=608)

        cv2.imshow('out_img', out_img)

        cv2.waitKey(0)

