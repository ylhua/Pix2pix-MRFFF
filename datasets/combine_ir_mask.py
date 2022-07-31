import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='/media/linger/udata/data_time/data/pm2/VIS')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='/media/linger/udata/data_time/data/pm2/TI')
parser.add_argument('--fold_C', dest='fold_C', help='input directory for image C', type=str, default='/media/linger/udata/data_time/data/pm2/mask')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='/media/linger/udata/data_time/data/pm2/ir_vis')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)', action='store_true')
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))
i = 1
while i == 1:
    img_fold_A = args.fold_A
    img_fold_B = args.fold_B
    # img_fold_C = args.fold_C
    img_list = os.listdir(img_fold_A)

    num_imgs = min(args.num_imgs, len(img_list))
    img_fold_AB = args.fold_AB
    if not os.path.isdir(img_fold_AB):
        os.makedirs(img_fold_AB)
    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)
        name_C = name_A.replace('.jpg', '.png')
        # path_C = os.path.join(img_fold_C, name_C)
        path_B = os.path.join(img_fold_B, name_A)
        if os.path.isfile(path_A):
            name_AB = name_C
            path_AB = os.path.join(img_fold_AB, name_AB)
            im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
            im_A = cv2.resize(im_A, (312, 312))
            im_B = cv2.imread(path_B, 1)
            im_B = cv2.resize(im_B, (312, 312))
            # im_C = cv2.imread(path_C, 1)
            # im_C = cv2.resize(im_C, (312, 312)), im_C
            im_AB = np.concatenate([im_A, im_B], 1)
            cv2.imwrite(path_AB, im_AB)
    i = i + 1