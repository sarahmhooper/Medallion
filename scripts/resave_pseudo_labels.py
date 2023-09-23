import csv
import os
import sys
import glob
import numpy as np
sys.path.append("../dauphin/image_segmentation/datasets")
from utils import custom_preproc_op


def resave(pred_seg_path, gt_seg_path, csv_path, n_classes):
    """
    Replace the segmentations in pred_seg_path with the ground truth segmentations in gt_seg_path specified in csv_path
    """

    if csv_path != 'None': # If a CSV is supplied, only copy the ground truth segmentations specified in the CSV
        
        # Get allowed GT labels
        with open(csv_path, newline="") as f:
            csv_data = list(csv.reader(f))
        csv_dict = {}
        for pid, sl_id in csv_data:
            if pid in csv_dict.keys():
                csv_dict[pid] += [sl_id]
            else:
                csv_dict[pid] = [sl_id]

        # Replace pseudo labels with GT labels, at the index specified in the CSV
        for key in csv_dict.keys():
            gt_path = os.path.join(gt_seg_path, key+'_seg.npy')
            prob_path = os.path.join(pred_seg_path, key+'_seg.npy')
            pred_path = os.path.join(pred_seg_path, key+'_binarized.npy')

            gt = np.load(gt_path)
            prob = np.load(prob_path)
            if os.path.exists(pred_path): pred = np.load(pred_path)
            gt = custom_preproc_op(gt, prob.shape[0], prob.shape[1], order=0)
            
            for cl_ind in range(n_classes):
                for idx in csv_dict[key]:
                    if os.path.exists(pred_path): 
                        pred[:,:,int(idx)] = gt[:,:,int(idx)]
                    slice_gt_seg = np.zeros_like(gt[:, :, int(idx)])
                    slice_gt_seg[gt[:, :, int(idx)] == cl_ind] = 1
                    prob[:, :, int(idx), int(cl_ind)] = slice_gt_seg

            np.save(prob_path, prob)
            if os.path.exists(pred_path): np.save(pred_path, pred)
            
    else: # If a CSV is not supplied, copy ground truth segmentations from all available seg masks
        
        for gt_path in glob.glob(os.path.join(gt_seg_path, '*_seg.npy')):
            key = gt_path.split('/')[-1].split('_seg.')[0]
            prob_path = os.path.join(pred_seg_path, key+'_seg.npy')
            pred_path = os.path.join(pred_seg_path, key+'_binarized.npy')

            gt = np.load(gt_path)
            prob = np.load(prob_path)
            if os.path.exists(pred_path): pred = np.load(pred_path)
            gt = custom_preproc_op(gt, prob.shape[0], prob.shape[1], order=0)
            
            for cl_ind in range(n_classes):
                for idx in range(gt.shape[-1]):
                    if os.path.exists(pred_path): 
                        pred[:,:,int(idx)] = gt[:,:,int(idx)]
                    slice_gt_seg = np.zeros_like(gt[:, :, int(idx)])
                    slice_gt_seg[gt[:, :, int(idx)] == cl_ind] = 1
                    prob[:, :, int(idx), int(cl_ind)] = slice_gt_seg

            np.save(prob_path, prob)
            if os.path.exists(pred_path): np.save(pred_path, pred)

if __name__ == "__main__":

    resave(sys.argv[1], sys.argv[2], sys.argv[3])
