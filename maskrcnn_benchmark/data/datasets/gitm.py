# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import os.path as op

import numpy as np
import torch

from .relation_tsv import RelationTSVDataset
from .evaluation.sg.box import bbox_overlaps
from .utils.label_loader import LabelLoader
from torch.utils.data import Dataset

import json
import cv2
from PIL import Image
import os.path as osp
import os
from maskrcnn_benchmark.structures.bounding_box import BoxList
import pickle as pkl


# VG data loader for Danfei Xu's Scene graph focused format.
# todo: if ordering of classes, attributes, relations changed
# todo make sure to re-write the obj_classes.txt/rel_classes.txt files


def cv2Img_to_Image(input_img):
    img = input_img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img

def get_unprocessed_img_ids(image_list,save_dir):
    feat_files=os.listdir(save_dir)
    exist_img_ids=[fn.split('.')[0] for fn in feat_files]
    unprocessed_img_ids=list(set(image_list)-set(exist_img_ids))
    return unprocessed_img_ids


def _box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations.
    If no overlapping boxes, use all of them."""
    overlaps = bbox_overlaps(boxes, boxes).numpy() > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes


class GITMDataset(Dataset):
    def __init__(self, yaml_file, extra_fields=(), transforms=None,
            is_load_label=True, force_box=False, force_box_path='', **kwargs):

        super(GITMDataset, self).__init__()

        # self.linelist_file
        # if 'train' in op.basename(self.linelist_file):
        #     self.split = "train"
        # elif 'test' in op.basename(self.linelist_file) \
        #         or 'val' in op.basename(self.linelist_file)\
        #         or 'valid' in op.basename(self.linelist_file):
        #     self.split = "test"
        # else:
        #     raise ValueError("Split must be one of [train, test], but get {}!".format(self.linelist_file))
        # filter_duplicate_rels = False and self.split == 'train'

        # if self.is_load_label:
        #     self.label_loader = LabelLoader(
        #         labelmap=self.labelmap,
        #         extra_fields=extra_fields,
        #         filter_duplicate_relations=filter_duplicate_rels,
        #         ignore_rel=["to the left of", "to the right of"],
        #     )

        # # get frequency prior for relations
        # if self.relation_on:
        #     self.freq_prior_file = op.splitext(self.label_file)[0] + ".freq_prior.npy"
        #     if self.split == 'train' and not op.exists(self.freq_prior_file):
        #         print("Computing frequency prior matrix...")
        #         fg_matrix, bg_matrix = self._get_freq_prior()
        #         prob_matrix = fg_matrix.astype(np.float32)
        #         prob_matrix[:, :, 0] = bg_matrix
        #         prob_matrix[:, :, 0] += 1
        #         prob_matrix /= np.sum(prob_matrix, 2)[:, :, None]
        #         np.save(self.freq_prior_file, prob_matrix)

        self.data_path = "/storage/data/wuyu1/refreasoning/data/images"
        # imagelist = os.listdir(self.data_path)
        with open('/public/home/weiyn1/coding/uniter/data_wyn/img_ids/img_ids_test.json','r') as f:
            imagelist = json.load(f)
        imagelist=['2378852']

        for_unprocessed=False
        if for_unprocessed:
            save_dir='/storage/group/hexm/weiyn/refreasoning/det_results/vinvl_with_fastrcnn_boxes/inference/vinvl_vg_x152c4/'
            image_list=get_unprocessed_img_ids(imagelist, save_dir)
        else:
            image_list = imagelist
        
        # image_list = list(imagelist.keys())[25000:]
        # random.seed(10)
        # random.shuffle(image_list)
        # image_list = imagelist

        print('num_image:', len(image_list))
        self.image_list = image_list
        self.transform = transforms
    
        self.force_box=force_box
        self.det_res_path=force_box_path

        if os.path.isfile(self.det_res_path):
            print(f'use boxes from {self.det_res_path}')
            with open(self.det_res_path,'rb') as f:
                self.det_rois=pkl.load(f)
        # self.det_res_path = '/storage/group/hexm/weiyn/refreasoning/det_results/faster_rcnn_x152_NMS0p3_th0p0_max100_min10'
    
    def __len__(self,):
        return len(self.image_list)

    # def __getitem__(self, index):

    #     img_name = self.image_list[index]
    #     # img_name = osp.basename(img_path)
    #     img = cv2.imread(osp.join(self.data_path, img_name))
    #     img_size = img.shape  # w, h
    #     img = cv2Img_to_Image(img)
    #     img, _ = self.transform(img, target=None)
    #     target=None

    #     return img, img_size, img_name
    def __getitem__(self, index):

        img_name = self.image_list[index]
        # img_name = osp.basename(img_path)
        img = cv2.imread(osp.join(self.data_path, f'{img_name}.jpg'))
        img_size = img.shape  # h,w,c
        img = cv2Img_to_Image(img)

        if self.force_box:
            if os.path.isdir(self.det_res_path):
                bboxes=np.load(osp.join(self.det_res_path,f'{img_name}.npz'))['bbox']
            else:
                bboxes=self.det_rois[img_name]['box']
            proposals=BoxList(bboxes, img.size, mode="xyxy")
        else:
            proposals=None

        # todo: box transform
        # img, _ = self.transform(img, target=None)
        img, proposals = self.transform(img, target=proposals)
       
        
        return img, proposals, img_size, img_name
    
    def _get_freq_prior(self, must_overlap=False):
        fg_matrix = np.zeros((
            len(self.class_to_ind),
            len(self.class_to_ind),
            len(self.relation_to_ind)
        ), dtype=np.int64)

        bg_matrix = np.zeros((
            len(self.class_to_ind),
            len(self.class_to_ind),
        ), dtype=np.int64)

        for ex_ind in range(self.__len__()):
            target = self.get_groundtruth(ex_ind)
            gt_classes = target.get_field('labels').numpy()
            gt_relations = target.get_field('relation_labels').numpy()
            gt_boxes = target.bbox

            # For the foreground, we'll just look at everything
            try:
                o1o2 = gt_classes[gt_relations[:, :2]]
                for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
                    fg_matrix[o1, o2, gtr] += 1

                # For the background, get all of the things that overlap.
                o1o2_total = gt_classes[np.array(
                    _box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
                for (o1, o2) in o1o2_total:
                    bg_matrix[o1, o2] += 1
            except IndexError as e:
                assert len(gt_relations) == 0

            if ex_ind % 20 == 0:
                print("processing {}/{}".format(ex_ind, self.__len__()))

        return fg_matrix, bg_matrix
    
    def get_groundtruth(self, idx, call=False):
        # similar to __getitem__ but without transform
        img = self.get_image(idx)
        img_size = img.size # w, h
        annotations = self.get_annotations(idx)
        target = self.get_target_from_annotations(annotations, img_size)
        if call:
            return img, target, annotations
        else:
            return target