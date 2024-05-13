import argparse
import cv2
import h5py
import numpy as np
import os
import tifffile
from glob import glob
from natsort.natsort import natsorted

from dataset_generator import crop_patch, crop_patch_wzfov


def gen_dataset(src_input_files, src_label_files, dst_path, date_index, splited_fov, if_mask):
    """
    generating datasets:
    input args: 
        src_input_files: input image files list, list[]
        src_label_files: label image files list, list[]
        dst_path: path for saving h5py file, str
    """
    # h5py file pathname, record the fov information
    h5py_path = dst_path + "/dataset_" + date_index + "_fov_" + \
                str(int(splited_fov[0] * 10)) + "_" + str(int(splited_fov[1] * 10)) + ".h5"
    h5f = h5py.File(h5py_path, 'w')

    for img_idx in range(len(src_input_files)):
        print("Now processing img pairs of %s", os.path.basename(src_input_files[img_idx]))
        img_input = tifffile.imread(src_input_files[img_idx])
        img_label = tifffile.imread(src_label_files[img_idx])

        if len(img_input.shape) == 2:
            img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
        if len(img_label.shape) == 2:
            img_label = cv2.cvtColor(img_label, cv2.COLOR_GRAY2RGB)
        
        img_input = cv2.resize(img_input, (400, 400))
        img_label = cv2.resize(img_label, (400, 400))

        # normalize the input and the label
        img_input = np.asarray(img_input / 255, np.float32)
        img_label = np.asarray(img_label / 255, np.float32)

        # concate input and label together
        img_pair = np.concatenate([img_input, img_label], 2)

        # crop the patch 
        if splited_fov == [0.0, 1.0]:
            patch_list = crop_patch(img_pair, 200, 200, False)
        else:
            patch_list = crop_patch_wzfov(img_pair, 200, 200, False, splited_fov, if_mask)

        # save the patches into h5py file
        for patch_idx in range(len(patch_list)):
            data = patch_list[patch_idx].copy()
            h5f.create_dataset(str(img_idx)+'_'+str(patch_idx), shape=(400,400,8), data=data)

    h5f.close()

def main():
    src_input_path = "/datasets/arglass/mydata/blurred/"
    src_label_path = "/datasets/arglass/mydata/sharp/"
    dst_path = "./datasets"
    date_ind = "xxxxxxxx"
    splited_fov = [0.0, 1.0]
    interval_idx = 0
    if_mask = True
    
    src_input_files = natsorted(glob(src_input_path + "/*.tiff"))[:1000]#[1000:1100]
    src_label_files = natsorted(glob(src_label_path + "/*.tiff"))[:1000]#[1000:1100]

    gen_dataset(src_input_files, src_label_files, dst_path, date_ind, \
                    [splited_fov[interval_idx], splited_fov[interval_idx+1]], if_mask)
    
    print("Done")


if __name__ == "__main__":
    main()
