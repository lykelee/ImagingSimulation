import os
import cv2
import time
import torch
import numpy as np
from glob import glob
from natsort.natsort import natsorted
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataloader import Dataset_from_h5
from deformable_unet import DFUNet
from loss import *
from option.option import args
from utils import *

from my_train import compare_psnr, forward, MyDataset, DataLoader, makedirs


def test():
    device = "cuda"

    blurred_path = "/datasets/arglass/testset/input"
    gt_path = "/datasets/arglass/testset/gt"
    #blurred_path = "/datasets/arglass/mydata/blurred"
    #gt_path = "/datasets/arglass/mydata/sharp"
    log_path = "./results/"
    
    imglog_dir = os.path.join(log_path, "images")

    makedirs(imglog_dir, mode=0o777, exist_ok=True)
    
    input_channel, output_channel = 3, 1
    model = DFUNet(input_channel, output_channel, args.n_channel, args.offset_channel).to(device)
    model.load_state_dict(torch.load("./logs/models/030.pth"))

    from lykutils.torchutils import parameter_count
    param_cnt = parameter_count(model)
    print(f"Parameter count: {param_cnt} = {param_cnt:.3e}")

    blurred_images = natsorted(glob(blurred_path + "/*.png"))
    gt_images = natsorted(glob(gt_path + "/*.png"))
    #blurred_images = natsorted(glob(blurred_path + "/*.tiff"))
    #gt_images = natsorted(glob(gt_path + "/*.tiff"))

    dataset = MyDataset(blurred_images, gt_images)

    # NOTE: We want to measure the inference time of one frame
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()

    psnr_sum = 0.0
    time_sum = 0.0
    cnt = 0

    def torch2numpy255(x):
        return np.clip(255 * x.detach().cpu().permute(1, 2, 0).numpy(), 0, 255).astype(np.uint8)
    
    with torch.no_grad():
        for blurred, gt in tqdm(dataloader):
            blurred, gt = blurred.to(device), gt.to(device)
            
            start_time = time.time()
            deblurred = forward(model, blurred, device)
            time_sum += (time.time() - start_time)

            psnr = compare_psnr(deblurred, gt)
            psnr_sum += psnr

            if cnt < 20:
                cv2.imwrite(os.path.join(imglog_dir, f"{cnt:0>6}_blurred.png"), torch2numpy255(blurred[0]))
                cv2.imwrite(os.path.join(imglog_dir, f"{cnt:0>6}_est.png"), torch2numpy255(deblurred[0]))
                cv2.imwrite(os.path.join(imglog_dir, f"{cnt:0>6}_gt.png"), torch2numpy255(gt[0]))

            cnt += 1

    psnr_avg = psnr_sum / cnt
    time_avg = time_sum / cnt
    
    print(f"avg psnr = {psnr_avg:.2f}, avg time = {time_avg:.3e}")
    
    print("Done")


def main():
    test()

if __name__ == "__main__":
    main()