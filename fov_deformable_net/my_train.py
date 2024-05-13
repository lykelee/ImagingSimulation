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


def compare_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100

    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

class MyDataset(Dataset):
    def __init__(self, blurred_files, gt_files):
        self.blurred_files = blurred_files
        self.gt_files = gt_files

    def __len__(self):
        return len(self.blurred_files)

    def __getitem__(self, idx):
        def read(file):
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (400, 400))
            img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(2).permute(2, 0, 1)

            return img
        
        blurred = read(self.blurred_files[idx])
        gt = read(self.gt_files[idx])
        
        return blurred, gt


def forward(model, blurred, device):
    b, _, h, w = blurred.shape

    yy, xx = torch.meshgrid(torch.linspace(0.0, 1.0, h, device=device), torch.linspace(0.0, 1.0, w, device=device))
    fov = torch.stack([yy, xx], dim=2).permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1)

    input = torch.cat([blurred, fov], dim=1)

    deblurred = model(input)

    return deblurred


def makedirs(name, mode=0o777, exist_ok=False):
    os.makedirs(name, mode=mode, exist_ok=exist_ok)
    os.chmod(name, mode=mode)
    

def train():
    device = "cuda"

    blurred_path = "/datasets/arglass/mydata/blurred"
    gt_path = "/datasets/arglass/mydata/sharp"
    log_path = "./logs"

    batch_size = 4

    ckpt_dir = os.path.join(log_path, "models")
    imglog_dir = os.path.join(log_path, "images")

    makedirs(log_path, mode=0o777, exist_ok=True)
    makedirs(ckpt_dir, mode=0o777, exist_ok=True)
    makedirs(imglog_dir, mode=0o777, exist_ok=True)

    train_size, test_size = 9600, 400

    blurred_images = natsorted(glob(blurred_path + "/*.png"))
    gt_images = natsorted(glob(gt_path + "/*.png"))
    train_blurred_images = blurred_images[:train_size]
    train_gt_images = gt_images[:train_size]
    test_blurred_images = blurred_images[train_size:train_size+test_size]
    test_gt_images = gt_images[train_size:train_size+test_size]

    train_dataset = MyDataset(train_blurred_images, train_gt_images)
    test_dataset = MyDataset(test_blurred_images, test_gt_images)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    input_channel, output_channel = 3, 1
    model = DFUNet(input_channel, output_channel, args.n_channel, args.offset_channel).to(device)
    model.initialize_weights()

    criterion = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #ccm = torch.from_numpy(np.ascontiguousarray(args.ccm)).float().cuda()

    tb_dir = os.path.join(log_path, "tb")
    writer = SummaryWriter(logdir=tb_dir)

    from lykutils.torchutils import parameter_count
    param_cnt = parameter_count(model)
    print(f"Parameter count: {param_cnt} = {param_cnt:.3e}")

    def torch2numpy255(x):
        return np.clip(255 * x.detach().cpu().permute(1, 2, 0).numpy(), 0, 255).astype(np.uint8)
    
    for epoch in range(1, 1000 + 1):
        loss_sum = 0.0
        cnt = 0
        for blurred, gt in tqdm(train_dataloader):
            blurred, gt = blurred.to(device), gt.to(device)

            deblurred = forward(model, blurred, device)

            loss = criterion(deblurred, gt)
            
            loss_sum += loss.item()
            cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        loss_avg = loss_sum / cnt
        writer.add_scalar("loss/train", loss_avg, epoch)
        writer.flush()
            
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{epoch:0>3}.pth"))

        if epoch % 1 == 0:
            psnr_sum = 0.0
            cnt = 0
            with torch.no_grad():
                for blurred, gt in tqdm(test_dataloader):
                    blurred, gt = blurred.to(device), gt.to(device)
                    
                    deblurred = forward(model, blurred, device)
                    
                    psnr = compare_psnr(deblurred, gt)
                    psnr_sum += psnr
                    cnt += 1
                
                psnr_avg = psnr_sum / cnt
            
            print(f"epoch {epoch}: avg psnr = {psnr_avg:.2f}")

            writer.add_scalar("psnr/validation", psnr_avg, epoch)
            writer.flush()

            cv2.imwrite(os.path.join(imglog_dir, f"{epoch:0>3}_blurred.png"), torch2numpy255(blurred[0]))
            cv2.imwrite(os.path.join(imglog_dir, f"{epoch:0>3}_est.png"), torch2numpy255(deblurred[0]))
            cv2.imwrite(os.path.join(imglog_dir, f"{epoch:0>3}_gt.png"), torch2numpy255(gt[0]))
    
    print("Done")

def main():
    train()

if __name__ == "__main__":
    main()