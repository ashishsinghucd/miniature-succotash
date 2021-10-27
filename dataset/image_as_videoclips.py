import numpy as np
import bisect
# import torch
# from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
from PIL import Image


# import cv2

def read_image_stack(path, start_i, end_i, size=None):
    img_stack = []
    for i in range(start_i, end_i, 1):
        # print(path+'{0:05d}.jpg'.format(i), '??')
        # img = cv2.imread(path+'{0:05d}.jpg'.format(i))
        img = Image.open(path + '/frame_{}.jpg'.format(i))
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is not None:
            # img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
            img = img.resize((size[1], size[0]))
        img = np.array(img)
        img_stack.append(img)
    img_stack = np.asarray(img_stack)
    # print(img_stack.shape)
    return img_stack


def read_image_stack_idx(path, idx, size=None):
    img_stack = []
    # idx = np.around(np.linspace(0, len-1, window)).astype(np.int32)
    for i in idx:
        img = Image.open(path + '/frame_{}.jpg'.format(i))
        if size is not None:
            img = img.resize((size[1], size[0]))
        img = np.array(img)
        img_stack.append(img)
    img_stack = np.asarray(img_stack)
    # print(img_stack.shape)
    return img_stack


class VideoClips(object):
    def __init__(self, vid_list, window, step, metadata, size=(192, 256), every_n_skip=None):
        self.video_list = vid_list

        if every_n_skip is None:
            every_n_skip = 1

        self.frame_list = self.get_resampled_frames(vid_list, metadata, window, step, every_n_skip)
        self.cumulative_sizes = np.asarray([len(n) for n in self.frame_list]).cumsum(0).tolist()
        self.every_n_skip = every_n_skip
        self.window = window
        self.size = size

    def get_clip(self, item):
        video_idx, clip_idx = self.get_clip_indx(item)
        v_id = self.video_list[video_idx]
        f_id = self.frame_list[video_idx][clip_idx]
        # last_clip_pts = self.frame_list[video_idx][-1]
        # print(v_id, f_id, last_clip_pts, item + 1 in self.cumulative_sizes)
        # print(video_idx, clip_idx, '!!!!', v_id, f_id, f_id+self.window-1)
        img_stack = read_image_stack_idx(v_id, f_id, self.size)
        # print(last_clip_pts, f_id)
        # is_last_clip = last_clip_pts == f_id
        is_last_clip = item + 1 in self.cumulative_sizes
        # print(total_frame_n, '!!!', v_id)
        return img_stack, v_id, video_idx, f_id, is_last_clip

    # print(video_idx, clip_idx, '!!!!', v_id, f_id, f_id+self.window-1)

    def get_clip_info(self, item):
        video_idx, clip_idx = self.get_clip_indx(item)
        v_id = self.video_list[video_idx]
        f_id = self.frame_list[video_idx][clip_idx]
        last_clip_pts = self.frame_list[video_idx][-1]
        is_last_clip = last_clip_pts == f_id
        return v_id, f_id, is_last_clip

    def __len__(self):
        return sum([len(n) for n in self.frame_list])

    def get_clip_indx(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
        return video_idx, clip_idx

    @staticmethod
    def get_resampled_frames(vid_list, metadata, window, step, every_n_skip):
        v_dict = {}
        sampled_frames_vid = []
        for v_id, v_pts in zip(metadata["video_paths"], metadata["video_pts"]):
            v_dict[v_id] = v_pts

        for v_id in vid_list:
            frame_list = v_dict[v_id]
            # print(v_id, frame_list, every_n_skip)
            frame_start_idx = range(0, len(frame_list) - window + 1, step)
            sampled_frame = [range(frame_list[idx], frame_list[idx] + window, every_n_skip) for idx in frame_start_idx]
            if len(sampled_frame) == 0:
                sampled_frame = [
                    np.around(np.linspace(0, len(frame_list) - 1, int(window / every_n_skip))).astype(np.int32)]
            sampled_frames_vid.append(sampled_frame)
            # print(frame_start_idx, sampled_frame)
            # continue
            '''
            frame_list = v_dict[v_id]
            frame_count.append(len(frame_list))
            frame_idx = range(0, len(frame_list), every_n_skip)
            sampled_frame = [frame_list[idx] for idx in frame_idx]
            #print(len(frame_list), len(sampled_frame))

            ####moving windows
            sampled_idx = range(0, len(sampled_frame)-window+1, step)
            sampled_frame = [sampled_frame[idx] for idx in sampled_idx]
            #print(len(frame_list), len(sampled_frame), '-'*20, sampled_frame)
            if len(sampled_frame) == 0:
                sampled_frame = [0]
            sampled_frames_vid.append(sampled_frame)
            '''
        # exit()
        return sampled_frames_vid

    def reduce_with_indx(self, indx):
        def select(d, j):
            return [x for i, x in enumerate(d) if i in j]

        self.frame_list = [select(x, y) for x, y in zip(self.frame_list, indx)]
        self.cumulative_sizes = np.asarray([len(n) for n in self.frame_list]).cumsum(0).tolist()
