import argparse
import datetime
import os
import time

import cv2

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation fps benchmark')
    parser.add_argument('--image_dir', type=str, default='./images')
    parser.add_argument('--resolution', type=str, default='720x480', help='network input resolution')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=4.0')
    args = parser.parse_args()

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    filenames = [f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
    images = [cv2.imread(f) for f in filenames]

    start_time = time.time()
    
    for img in range(len(images)):
        humans = e.inference(images[img], resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

    print('Average FPS:', len(images) / (time.time() - start_time))

