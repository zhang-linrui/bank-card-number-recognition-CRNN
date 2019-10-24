# -*- coding: utf-8 -*-


import tensorflow as tf

# ctpn
from ctpn.nets.model_train import sess, input_image, input_im_info, bbox_pred, cls_prob
from ctpn.utils.rpn_msr.proposal_layer import proposal_layer
from ctpn.utils.text_connector.detectors import TextDetector

# bank card crnn
import crnn.vgg_blstm_ctc as vgg_blstm_ctc
from crnn.bankcard_predict import single_recognition

import numpy as np
import cv2
import os


def use_ctpn_net(img, filename):
    h0, w0, c0 = img.shape
    if h0 > w0:
        w0 = int(1.0 * 640 * w0 / h0)
        h0 = 640
    else:
        h0 = int(1.0 * 640 * h0 / w0)
        w0 = 640
    roi = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA)
    h, w, c = roi.shape
    im_info = np.array([h, w, c]).reshape([1, 3])
    bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                           feed_dict={input_image: [roi],
                                                      input_im_info: im_info})
    textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
    scores = textsegs[:, 0]
    textsegs = textsegs[:, 1:5]
    textdetector = TextDetector(DETECT_MODE='O')
    boxes = textdetector.detect(textsegs, scores[:, np.newaxis], roi.shape[:2])  # 通过调参或换模型可优化效率效果
    try:
        box = boxes[0]
        #box[2] += 10
        #box[4] += 10
        pts1 = np.float32([[box[0], box[1]], [box[2], box[3]], [box[6], box[7]], [box[4], box[5]]])
        pts2 = np.float32([[0, 0], [256, 0], [0, 32], [256, 32]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        image_dst = cv2.warpPerspective(roi, M, (256, 32))
        cv2.imwrite('res_detection/' + filename, image_dst)
        res = single_recognition(image_dst)
        print res
        cv2.polylines(roi, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(roi, res, (int(box[0]), int(box[1]) - 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite('res_recognition/' + filename, roi)
    except:
        print "fail to locate target"

datapath = 'data/'
files = os.listdir('data/')
for file in files:
    img = cv2.imread(datapath + file)
    use_ctpn_net(img, file)
    print(file + '--OK')
