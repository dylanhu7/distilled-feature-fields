import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils import data
import encoding.utils as utils
from encoding.datasets import test_batchify_fn 
from encoding.models.sseg import BaseNet
from modules.lseg_module import LSegModule
#from utils import Resize
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from data import get_original_dataset
from additional_utils.encoding_models import MultiEvalModule as LSeg_MultiEvalModule
import sklearn
import sklearn.decomposition
import time
from segment_anything import sam_model_registry, SamPredictor
import glob
import cv2

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description="PyTorch Segmentation")
        # model and dataset
        parser.add_argument(
            "--model", type=str, default="encnet", help="model name (default: encnet)"
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="clip_vitl16_384",
            help="backbone name (default: resnet50)",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            # default="ade20k",
            default="ignore",
            help="dataset name (default: pascal12)",
        )
        parser.add_argument(
            "--workers", type=int, default=16, metavar="N", help="dataloader threads"
        )
        parser.add_argument(
            "--base-size", type=int, default=520, help="base image size"
        )
        parser.add_argument(
            "--crop-size", type=int, default=480, help="crop image size"
        )
        parser.add_argument(
            "--train-split",
            type=str,
            default="train",
            help="dataset train split (default: train)",
        )
        # training hyper params
        parser.add_argument(
            "--aux", action="store_true", default=False, help="Auxilary Loss"
        )
        parser.add_argument(
            "--se-loss",
            action="store_true",
            default=False,
            help="Semantic Encoding Loss SE-loss",
        )
        parser.add_argument(
            "--se-weight", type=float, default=0.2, help="SE-loss weight (default: 0.2)"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                            training (default: auto)",
        )
        parser.add_argument(
            "--test-batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                            testing (default: same as batch size)",
        )
        # cuda, seed and logging
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            default=False,
            help="disables CUDA training",
        )
        parser.add_argument(
            "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
        )
        parser.add_argument(
            "--weights", type=str, default=None, help="checkpoint to test"
        )
        parser.add_argument(
            "--eval", action="store_true", default=False, help="evaluating mIoU"
        )
        parser.add_argument(
            "--export",
            type=str,
            default=None,
            help="put the path to resuming file if needed",
        )

        parser.add_argument(
            "--acc-bn",
            action="store_true",
            default=False,
            help="Re-accumulate BN statistics",
        )
        parser.add_argument(
            "--test-val",
            action="store_true",
            default=False,
            help="generate masks on val set",
        )
        parser.add_argument(
            "--no-val",
            action="store_true",
            default=False,
            help="skip validation during training",
        )
        parser.add_argument(
            "--module",
            default='lseg',
            help="select model definition",
        )
        # test option
        parser.add_argument(
            "--data-path", type=str, default=None, help="path to test image folder"
        )
        parser.add_argument(
            "--no-scaleinv",
            dest="scale_inv",
            default=True,
            action="store_false",
            help="turn off scaleinv layers",
        )
        parser.add_argument(
            "--widehead", default=False, action="store_true", help="wider output head"
        )
        parser.add_argument(
            "--widehead_hr",
            default=False,
            action="store_true",
            help="wider output head",
        )
        parser.add_argument(
            "--ignore_index",
            type=int,
            default=-1,
            help="numeric value of ignore label in gt",
        )
        parser.add_argument(
            "--label_src",
            type=str,
            default="default",
            help="how to get the labels",
        )
        parser.add_argument(
            "--jobname",
            type=str,
            default="default",
            help="select which dataset",
        )
        parser.add_argument(
            "--no-strict",
            dest="strict",
            default=True,
            action="store_false",
            help="no-strict copy the model",
        )
        parser.add_argument(
            "--arch_option",
            type=int,
            default=0,
            help="which kind of architecture to be used",
        )
        parser.add_argument(
            "--block_depth",
            type=int,
            default=0,
            help="how many blocks should be used",
        )
        parser.add_argument(
            "--activation",
            choices=['lrelu', 'tanh'],
            default="lrelu",
            help="use which activation to activate the block",
        )

        parser.add_argument(
            "--outdir",
            default="outdir_ours",
            help="output direcory of features",
        )
        parser.add_argument(
            "--test-rgb-dir",
            help="test rgb dir",
            required=True,
        )

        parser.add_argument(
            "--resize-max", type=float, default=1.25, help=""
        )

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args


    
def encode_images(args):
    #print working directory
    print(os.getcwd())
    os.makedirs(args.outdir, exist_ok=True)
    sam_checkpoint = "../../../segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = torch.device("cuda")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    # load image for each image in args.test_rgb_dir
    for rgb_path in tqdm(glob.glob(os.path.join(args.test_rgb_dir, "*.png"))):
        name = os.path.basename(rgb_path).split(".")[0]
        bgr = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(os.path.join(args.outdir, name + ".png"), bgr)
        rgb = cv2.cvtColor(cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)
        embedding = predictor.get_image_embedding() # [1, 256, 64, 64] (batch, channels, height, width)
        embedding = embedding.squeeze(0).cpu().numpy()
        embedding_path = os.path.join(args.outdir, name + ".pth")
        torch.save(embedding, embedding_path)

if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count() 
    # test(args)
    encode_images(args)
