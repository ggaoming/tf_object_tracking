# -*- coding:utf-8 -*
"""
@version: 0.0
@Author by Ggao
@Mail: ggao_liming@qq.com
@File: votdataset.py
@time: 2017-07-07
"""
import os
import cv2
import numpy as np
import random

class vot(object):
    def __init__(self):
        self.input_path = 'Dataset/vot'
        self.output_img = 'Dataset/vot/vot'
        self.output_path = 'data_set/vot.txt'
        self.input_list = self.load_path(self.input_path)
        print len(self.input_list)
        pass

    def is_root_file(self, p_list):
        for p in p_list:
            if '.jpg' in p or '.png' in p:
                return True
        return False

    def load_path(self, path):
        assert os.path.exists(path)
        if not os.path.isdir(path):
            return []
        p_list = os.listdir(path)
        if self.is_root_file(p_list):
            return [path]
        else:
            res = []
            for p in p_list:
                res += self.load_path(os.path.join(path, p))
            return res
        pass

    @staticmethod
    def get_sub_img(img, x0, y0, x1, y1):
        img_h, img_w, _ = img.shape
        w = x1 - x0
        h = y1 - y0
        if not (w > 0 and h > 0):
            return None
        start_x = x0 - w / 2
        start_x_ = max(0, 0 - start_x)  # for new img start x
        start_x = max(0, start_x)  # for img start x

        start_y = y0 - h / 2
        start_y_ = max(0, 0 - start_y)
        start_y = max(0, start_y)

        end_x = x1 + w / 2
        end_x = min(img_w, end_x)  # for img end x

        end_y = y1 + h / 2
        end_y = min(img_h, end_y)  # for img end y

        new_img = np.zeros((2*h, 2*w, 3), dtype=np.uint8)

        sub_img = img[start_y:end_y, start_x:end_x]
        new_h, new_w, _ = sub_img.shape
        new_img[start_y_: start_y_+new_h, start_x_:start_x_+new_w] = sub_img
        return new_img
        pass

    def generate_time_step(self, step=2):
        # random select input sequence
        vot_path = random.sample(self.input_list, 1)[0]
        # images and annotation
        img_files = os.listdir(vot_path)
        img_files.sort()
        img_files = filter(lambda x: '.jpg' in x and x[0] != '.', img_files)
        anno_file = os.path.join(vot_path, 'groundtruth.txt')
        anno_lines = open(anno_file).readlines()
        time_line = min(len(img_files), len(anno_lines))
        # start_time to end_time
        end_time = time_line - step
        start_time = random.randint(0, end_time)
        end_time = start_time+step
        assert end_time <= time_line
        img_files = [os.path.join(vot_path, img_files[t]) for t in range(start_time, end_time)]
        anno_lines = [anno_lines[t] for t in range(start_time, end_time)]
        org_x0, org_x1, org_y0, org_y1 = None, None, None, None  # information in pre step
        org_img = None
        X0 = []  # save by time line
        X1 = []
        Y = []
        for i, a in enumerate(anno_lines):
            a = a.strip().split(',')
            a = [int(eval(tt)) for tt in a]
            # 01, 23, 45, 67
            img = cv2.imread(img_files[i])
            x0, x1 = min(a[0], a[2], a[4], a[6]), max(a[0], a[2], a[4], a[6])
            y0, y1 = min(a[1], a[3], a[5], a[7]), max(a[1], a[3], a[5], a[7])
            if org_img is not None:
                # add random changes
                diff_scale = 0.05
                o_w = int((org_x1 - org_x0)*diff_scale)
                o_h = int((org_y1 - org_y1)*diff_scale)
                org_x0 += random.randint(-o_w, o_w)
                org_x1 += random.randint(-o_w, o_w)
                org_y0 += random.randint(-o_h, o_h)
                org_y1 += random.randint(-o_h, o_h)
                sub_org_img = self.get_sub_img(org_img, org_x0, org_y0, org_x1, org_y1)
                sub_img = self.get_sub_img(img, org_x0, org_y0, org_x1, org_y1)
                if sub_img is None or sub_org_img is None:
                    print 'Error zero size', img_files[i]
                    return self.generate_time_step(step=step)
                X0.append(sub_org_img)
                X1.append(sub_img)
                Y.append([org_x0+0, org_x1+0, org_y0+0, org_y1+0, x0+0, x1+0, y0+0, y1+0])
                pass
            org_img = img.copy()
            org_x0, org_x1, org_y0, org_y1 = x0, x1, y0, y1
        if start_time == 0:
            # special operation for first init frame
            repeat_time = random.randint(1, step-1)
            X0 = [X0[0]]*repeat_time + X0[repeat_time:]
            X1 = [X1[0]]*repeat_time + X1[repeat_time:]
            Y[-1] = Y[-repeat_time]
            pass
        return X0, X1, Y[-1]
        pass

    def next_batch(self, batch_size, time_step):
        X0 = []
        X1 = []
        Y = []
        for _ in range(batch_size):
            X0_, X1_, Y_ = self.generate_time_step(step=time_step)
            X0_ = [cv2.resize(i, (227, 227)) for i in X0_]
            X1_ = [cv2.resize(i, (227, 227)) for i in X1_]
            x00, x01, y00, y01, x10, x11, y10, y11 = Y_
            w0, h0 = x01 - x00, y01 - y00
            dx0 = (x10 - (x00 - w0 * 0.5)) * 1.0 / (2 * w0)
            dx1 = (x11 - (x00 - w0 * 0.5)) * 1.0 / (2 * w0)
            dy0 = (y10 - (y00 - h0 * 0.5)) * 1.0 / (2 * h0)
            dy1 = (y11 - (y00 - h0 * 0.5)) * 1.0 / (2 * h0)
            X0.append(X0_)
            X1.append(X1_)
            Y.append([dx0, dx1, dy0, dy1])
            pass
        return X0, X1, Y
        pass


if __name__ == '__main__':
    app = vot()
    for i in range(1000):
        app.next_batch(10, 3)