import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import numpy
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from dataset import Dataset
import AAcUNet
from metrics import di_co, batch, mean, iou_sc ,ppv,sen
import losses
from utils import str2bool, count_params
import joblib
from hausdorff import hausdorff_distance
import imageio
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='AACUnet',
                        help='model name')
    parser.add_argument('--mode', default="Calculate")
    args = parser.parse_args()
    return args
def main():
    val_args = parse_args()
    args = joblib.load('models/%s/args.pkl' %val_args.name)
    print("args:", args)
    if not os.path.exists('output/%s' %args.name):
        os.makedirs('output/%s' %args.name)
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')
    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    print("=> creating model %s" %args.arch)
    model = AAcUNet.AAcUNet()
    model = model.cuda()
    img_paths = glob(r'*')
    mask_paths = glob(r'*')
    val_img_paths = img_paths
    val_mask_paths = mask_paths
    model.load_state_dict(torch.load('models/%s/model.pth' %args.name))
    model.eval()
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    if val_args.mode == "Calculate":
        print("val_args.mode == Calculate")
        wt_dices = []
        tc_dices = []
        et_dices = []
        wt_sensitivities = []
        tc_sensitivities = []
        et_sensitivities = []
        wt_ppvs = []
        tc_ppvs = []
        et_ppvs = []
        wt_Hausdorf = []
        tc_Hausdorf = []
        et_Hausdorf = []
        wtMaskList = []
        tcMaskList = []
        etMaskList = []
        wtPbList = []
        tcPbList = []
        etPbList = []
        maskPath = glob("output/%s/" % args.name + "GT/*.png")
        pbPath = glob("output/%s/" % args.name + "/*.png")
        print("output/%s/" % args.name + "/*.png")
        print("output/%s/" % args.name + "/*.png", )
        if len(maskPath) == 0:
            return
        for myi in tqdm(range(len(maskPath))):
            mask = imread(maskPath[myi])
            pb = imread(pbPath[myi])
            wtmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            wtpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            tcmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            tcpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            etmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            etpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            for idx in range(mask.shape[0]):
                for idy in range(mask.shape[1]):
                    if mask[idx, idy, :].any() != 0:
                        wtmaskregion[idx, idy] = 1
                    if pb[idx, idy, :].any() != 0:
                        wtpbregion[idx, idy] = 1
                    if mask[idx, idy, 0] == 255:
                        tcmaskregion[idx, idy] = 1
                    if pb[idx, idy, 0] == 255:
                        tcpbregion[idx, idy] = 1
                    if mask[idx, idy, 1] == 128:
                        etmaskregion[idx, idy] = 1
                    if pb[idx, idy, 1] == 128:
                        etpbregion[idx, idy] = 1
            dice = di_co(wtpbregion,wtmaskregion)
            wt_dices.append(dice)
            ppv_n = ppv(wtpbregion, wtmaskregion)
            wt_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(wtmaskregion, wtpbregion)
            wt_Hausdorf.append(Hausdorff)
            sen_n = sen(wtpbregion, wtmaskregion)
            wt_sensitivities.append(sen_n)
            dice = di_co(tcpbregion, tcmaskregion)
            tc_dices.append(dice)
            ppv_n = ppv(tcpbregion, tcmaskregion)
            tc_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(tcmaskregion, tcpbregion)
            tc_Hausdorf.append(Hausdorff)
            sen_n = sen(tcpbregion, tcmaskregion)
            tc_sensitivities.append(sen_n)
            dice = di_co(etpbregion, etmaskregion)
            et_dices.append(dice)
            ppv_n = ppv(etpbregion, etmaskregion)
            et_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(etmaskregion, etpbregion)
            et_Hausdorf.append(Hausdorff)
            sen_n = sen(etpbregion, etmaskregion)
            et_sensitivities.append(sen_n)
        print('WT Dice: %.4f' % np.mean(wt_dices))
        print('TC Dice: %.4f' % np.mean(tc_dices))
        print('ET Dice: %.4f' % np.mean(et_dices))
        print("=============")
        print('WT PPV: %.4f' % np.mean(wt_ppvs))
        print('TC PPV: %.4f' % np.mean(tc_ppvs))
        print('ET PPV: %.4f' % np.mean(et_ppvs))
        print("=============")
        print('WT sen: %.4f' % np.mean(wt_sensitivities))
        print('TC sen: %.4f' % np.mean(tc_sensitivities))
        print('ET sen: %.4f' % np.mean(et_sensitivities))
        print("=============")
        print('WT Hausdorff: %.4f' % np.mean(wt_Hausdorf))
        print('TC Hausdorff: %.4f' % np.mean(tc_Hausdorf))
        print('ET Hausdorff: %.4f' % np.mean(et_Hausdorf))
        print("=============")
if __name__ == '__main__':
    main( )
