# -*- coding:utf-8 -*
"""
@version: 0.0
@Author by Ggao
@Mail: ggao_liming@qq.com
@File: imgnetdataset.py
@time: 2017-07-08
"""
import cv2
import numpy as np
import random
import os
import logging
import pickle
from votdataset import vot

logging.basicConfig(level=logging.INFO)


class img_net(object):
    def __init__(self):
        self.img_path = ['Dataset/ImageNet 256x256/train_pic256x256',
                         'Dataset/ImageNet 256x256/val_pic256x256']
        self.img_path = self.load_path()
        line_pkl_path = 'data_set/estimate_lines.pkl'
        self.estimate_line = pickle.load(open(line_pkl_path))
        logging.info('img count {0}'.format(len(self.img_path)))

    def load_path(self):
        res = []
        pkl_path = 'data_set/imgnet.pkl'
        if os.path.exists(pkl_path):
            return pickle.load(open(pkl_path))
        for p_ in self.img_path:
            p_list = os.listdir(p_)
            p_list = filter(lambda x: ('.jpg' in x or '.png' in x or '.JPEG' in x) and x[0] != '.', p_list)
            p_list = [os.path.join(p_, p) for p in p_list]
            res += p_list
        pickle.dump(res, open(pkl_path, 'w'))
        return res

    def add_img(self, background_img, target_img, x, y):
        bh, bw, _ = background_img.shape
        th, tw, _ = target_img.shape
        # img location center x,y in background image
        x = x*1.0/500*bw
        y = y*1.0/500*bw
        # the left top and right bottom x,y
        tlx = int(x - tw*0.5)  # less than zero
        tly = int(y - th*0.5)
        brx = int(x + tw*0.5)  # bigger than background image
        bry = int(y + th*0.5)
        # cut the value out of the background image
        dtlx = max(0, 1-tlx)  # tl part
        dtly = max(0, 1-tly)
        dbrx = max(0, brx - bw + 1)  # br part
        dbry = max(0, bry - bh + 1)
        # new location in background
        tlx = max(0, tlx)
        tly = max(0, tly)
        target_img = target_img[dtly: th-dbry, dtlx: tw-dbrx]
        nh, nw, _ = target_img.shape
        background_img[tly:tly+nh, tlx:tlx+nw] = target_img
        return tlx, tlx+nw, tly, tly+nh
        pass

    def get_info(self, time_step=100):
        object_num = random.randint(3, 7)
        img_list = random.sample(self.img_path, object_num)
        background_img = cv2.imread(img_list[0])
        img_list = img_list[1:]
        # targets'id
        target_index = random.randint(0, len(img_list)-1)
        object_lines = []
        object_scales = []
        for i in range(len(img_list)):
            # target's speed
            l_step = random.randint(1, 4)
            # target's scale
            object_scales.append([random.randint(30, 60), random.randint(30, 60)])
            tmp_start = random.randint(0, len(self.estimate_line)/l_step - 1 - time_step)
            tmp_line = self.estimate_line[::l_step][tmp_start: tmp_start+time_step]
            # target's line
            object_lines.append(tmp_line[:])
            pass
        target_imgs = [cv2.imread(i) for i in img_list]
        org_img = None
        org_x0, org_x1, org_y0, org_y1 = None, None, None, None
        X0 = []  # save by time line
        X1 = []
        Y = []
        x0, y0, x1, y1 = 0, 0, 0, 0
        for t in range(time_step):
            b_img = background_img.copy()
            for i in range(len(target_imgs)):
                sub_img = cv2.resize(target_imgs[i], tuple(object_scales[i]))
                sub_location = object_lines[i][t]
                x0_, x1_, y0_, y1_ = self.add_img(b_img, sub_img, sub_location[0], sub_location[1])
                if i == target_index:
                    x0, x1, y0, y1 = x0_, x1_, y0_, y1_
            if org_img is not None:
                # random changes
                diff_scale = 0.1
                o_w = int((org_x1 - org_x0) * diff_scale)
                o_h = int((org_y1 - org_y1) * diff_scale)
                org_x0 += random.randint(-o_w, o_w)
                org_x1 += random.randint(-o_w, o_w)
                org_y0 += random.randint(-o_h, o_h)
                org_y1 += random.randint(-o_h, o_h)
                sub_org = vot.get_sub_img(org_img, org_x0, org_y0, org_x1, org_y1)
                sub_img = vot.get_sub_img(b_img, org_x0, org_y0, org_x1, org_y1)
                X0.append(sub_org)
                X1.append(sub_img)
                Y.append([org_x0 + 0, org_x1 + 0, org_y0 + 0, org_y1 + 0, x0 + 0, x1 + 0, y0 + 0, y1 + 0])
                pass
            org_img = b_img.copy()
            org_x0, org_x1, org_y0, org_y1 = x0, x1, y0, y1
        return X0, X1, Y[-1]

    def get_batch(self, batch_size=32, time_step=2):
        X0 = []
        X1 = []
        Y = []
        for _ in range(batch_size):
            X0_, X1_, Y_ = self.get_info(time_step=time_step)
            X0_ = [cv2.resize(i, (227, 227)) for i in X0_]
            X1_ = [cv2.resize(i, (227, 227)) for i in X1_]
            x00, x01, y00, y01, x10, x11, y10, y11 = Y_
            w0, h0 = x01 - x00, y01 - y00
            # targets's position to estimate
            dx0 = (x10 - (x00 - w0 * 0.5)) * 1.0 / (2 * w0)
            dx1 = (x11 - (x00 - w0 * 0.5)) * 1.0 / (2 * w0)
            dy0 = (y10 - (y00 - h0 * 0.5)) * 1.0 / (2 * h0)
            dy1 = (y11 - (y00 - h0 * 0.5)) * 1.0 / (2 * h0)
            # dx0 = (x10 - x00) * 1.0 / (2 * w0)
            # dx1 = (x11 - x00) * 1.0 / (2 * w0)
            # dy0 = (y10 - y00) * 1.0 / (2 * h0)
            # dy1 = (y11 - y00) * 1.0 / (2 * h0)
            X0.append(X0_)
            X1.append(X1_)
            Y.append([dx0, dx1, dy0, dy1])
            pass
        return X0, X1, Y
        pass

if __name__ == '__main__':
    app = img_net()
    app.get_info()
    # for i in range(1000):
    #     app.get_batch()
    # pass
