import os
import torch
from torch.utils.data import Dataset
from dataset.image_as_videoclips import VideoClips

from torchvision import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo, NormalizeVideo, RandomHorizontalFlipVideo
import csv
from slowfast.datasets.utils import pack_pathway_output, spatial_sampling


def spatial_sample(img_stack_norm, spatial_idx, size_config, random_flip, inv_sampling):
    img_stack = spatial_sampling(
        img_stack_norm,
        spatial_idx=spatial_idx,
        min_scale=size_config[0],
        max_scale=size_config[1],
        crop_size=size_config[2],
        random_horizontal_flip=random_flip,
        inverse_uniform_sampling=inv_sampling,
    )
    return img_stack


class MP_dataset(Dataset):
    def __init__(self, mode, window, step, data_path, size=(192, 256), every_n_skip=None, cfg=None):
        self.datapath = data_path
        video_list, self.video_label_list = self.get_video_list(self.datapath, mode)
        label_indx = ['A', 'N', 'Arch', 'R']
        self.video_label_list = [label_indx.index(l) for l in self.video_label_list]
        video_list, metadata, gt_ids = self.get_video_meta(self.datapath, video_list, mode)

        self.video_list = VideoClips(video_list, window, step, metadata, size, every_n_skip=every_n_skip)
        if mode == "train":
            self.transforms = T.Compose([ToTensorVideo(), NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])])
            self.spatial_idx = -1
            self.size_config = (256, 256, 256)
            # self.size_config = (cfg.DATA.TRAIN_JITTER_SCALES[0], cfg.DATA.TRAIN_JITTER_SCALES[1],
            #                     cfg.DATA.TRAIN_CROP_SIZE)  # (356, 446, 312)#(256, 320, 224)
            self.random_flip = cfg.DATA.RANDOM_FLIP
        elif mode == 'val':
            self.transforms = T.Compose([ToTensorVideo(), NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])])
            self.spatial_idx = -1
            self.size_config = (256, 256, 256)
            # self.size_config = (cfg.DATA.TRAIN_JITTER_SCALES[0], cfg.DATA.TRAIN_JITTER_SCALES[1],
            #                     cfg.DATA.TRAIN_CROP_SIZE)  # (356, 446, 312)#(256, 320, 224)
            self.random_flip = cfg.DATA.RANDOM_FLIP
        elif mode == 'test':
            self.transforms = T.Compose([ToTensorVideo(), NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])])
            self.spatial_idx = None
            self.size_config = (cfg.DATA.TEST_CROP_SIZE, cfg.DATA.TEST_CROP_SIZE,
                                cfg.DATA.TEST_CROP_SIZE)  # (356, 356, 356) #(256, 256, 256)
            self.random_flip = False  # cfg.DATA.RANDOM_FLIP

        self.window = window
        self.cfg = cfg
        self.mode = mode
        self.every_n_skip = every_n_skip
        self.size = size
        self.step = step
        # self.min_max_scale_train = (256, 320)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        # print(self.box_list[index].shape)
        img_stack, v_id, v_idx, f_id, is_last_clip = self.video_list.get_clip(index)
        img_stack = torch.from_numpy(img_stack)
        img_stack_norm = self.transforms(img_stack)
        # exit()
        if self.mode == 'train' or self.mode == 'val':
            img_stack_norm = spatial_sample(img_stack_norm, self.spatial_idx, self.size_config, self.random_flip,
                                            self.cfg.DATA.INV_UNIFORM_SAMPLE)
            img_stack = pack_pathway_output(self.cfg, img_stack_norm)
        elif self.mode == 'test':
            img_stack = None
            for i in range(3):
                img_stack_tmp = spatial_sample(img_stack_norm, i, self.size_config, False,
                                               self.cfg.DATA.INV_UNIFORM_SAMPLE)
                img_stack_tmp = pack_pathway_output(self.cfg, img_stack_tmp)
                if img_stack is None:
                    img_stack = [[x] for x in img_stack_tmp]
                else:
                    for j, x in enumerate(img_stack_tmp):
                        img_stack[j].append(x)
            img_stack = [torch.stack(x) for x in img_stack]

        # for y in img_stack:print(y.shape, '??')
        # exit()
        label = self.video_label_list[v_idx]
        return img_stack, v_id, f_id, label, is_last_clip
        return img_stack_norm, v_id, f_id, label, is_last_clip
        return img_stack_norm, img_stack, v_id, v_idx, label, is_last_clip

    @staticmethod
    def get_video_list(datapath, mode, random_seed=10, n_val=4):
        import numpy as np
        np.random.seed(random_seed)

        # arr = np.arange(1452)
        # np.random.shuffle(arr)
        def readcsv(label_path):
            all_data = []
            with open(label_path, newline='') as csvfile:
                label_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                next(label_reader)
                for row in label_reader:
                    all_data.append(row)
            return list(map(list, zip(*all_data)))

        def get_uniq(vid_list):
            uniq_idx, uniq_list = [], []
            for i, x in enumerate(vid_list):
                x_uniq = x.split('_')
                x_uniq = x_uniq[0]  # + '_' + x_uniq[1]
                if x_uniq not in uniq_list:
                    uniq_list.append(x_uniq)
                    uniq_idx.append([i])
                else:
                    tmp_idx = uniq_list.index(x_uniq)
                    uniq_idx[tmp_idx].append(i)
            return uniq_idx

        # select_i = [0, 12, 24, 36]
        # select_i = [0, 12, 24, 36]
        if mode == 'train':
            label_path = '{}train.csv'.format(datapath)
            [vid_list, vid_label_list] = readcsv(label_path)
            # uniq_idx = get_uniq(vid_list)
            # select_i = np.random.choice(len(uniq_idx), n_val)  # val2 is 7 val is 4?
            # print(select_i)
            # arr = [y for i, x in enumerate(uniq_idx) if i not in select_i for y in x]
            #
            # vid_list = [vid_list[tmp] for tmp in arr]
            # vid_label_list = [vid_label_list[tmp] for tmp in arr]
            # print(len(vid_list), len(vid_label_list))

        elif mode == 'val':
            label_path = '{}val.csv'.format(datapath)
            [vid_list, vid_label_list] = readcsv(label_path)
            # uniq_idx = get_uniq(vid_list)
            # select_i = np.random.choice(len(uniq_idx), n_val)
            # print(select_i)
            # arr = [y for i, x in enumerate(uniq_idx) if i in select_i for y in x]
            # # arr = [x[y] for x in uniq_idx for y in np.random.choice(len(x), 2)]
            #
            # vid_list = [vid_list[tmp] for tmp in arr]
            # vid_label_list = [vid_label_list[tmp] for tmp in arr]
            # print(len(vid_list), len(vid_label_list))
            # exit()

        elif mode == 'test':
            label_path = '{}test.csv'.format(datapath)
            [vid_list, vid_label_list] = readcsv(label_path)
        return vid_list, vid_label_list

    def reinit(self, random_seed, n_val):
        video_list, self.video_label_list = self.get_video_list(self.datapath, self.mode, random_seed, n_val)
        label_indx = ['A', 'N', 'Arch', 'R']
        self.video_label_list = [label_indx.index(l) for l in self.video_label_list]
        video_list, metadata, gt_ids = self.get_video_meta(self.datapath, video_list, self.mode)
        self.video_list = VideoClips(video_list, self.window, self.step, metadata, self.size,
                                     every_n_skip=self.every_n_skip)

    @staticmethod
    def get_video_meta(datapath, video_list, mode):
        video_pts = []
        gt_ids = []
        full_pathid = []
        for video_id in video_list:
            # print('{}MP/{}'.format(datapath, video_id))
            # exit()
            full_pathid.append('{}MP/{}'.format(datapath, video_id))
            frame_list = [x for x in os.listdir('{}MP/{}'.format(datapath, video_id)) if '.jpg' in x]
            # print(len(frame_list))
            # frame_list.sort()
            new_frame_list = range(len(frame_list))
            video_pts.append(new_frame_list)
        metadata = {
            "video_paths": full_pathid,
            "video_pts": video_pts,
        }
        return full_pathid, metadata, gt_ids
