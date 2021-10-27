from torch.utils.data import Sampler
import torch


class UniformClipSampler(Sampler):
    def __init__(self, video_clips, num_clips_per_video, shuffle=True):
        self.video_clips = video_clips
        self.num_clips_per_video = num_clips_per_video
        self.shuffle = shuffle

    def __iter__(self):
        idxs = []
        s = 0
        # select num_clips_per_video for each video, uniformly spaced
        for c in self.video_clips.frame_list:
            length = len(c)
            if length == 0:
                # corner case where video decoding fails
                continue

            sampled = (
                torch.linspace(s, s + length - 1, steps=self.num_clips_per_video)
                    .floor()
                    .to(torch.int64)
            )
            s += length
            idxs.append(sampled)
        idxs_ = torch.cat(idxs)
        # print(idxs_)
        # exit()
        # shuffle all clips randomly
        if self.shuffle:
            perm = torch.randperm(len(idxs_))
            idxs = idxs_[perm]
        else:
            idxs = idxs_
        return iter(idxs.tolist())

    def __len__(self) -> int:
        return sum(
            self.num_clips_per_video for c in self.video_clips.frame_list if len(c) > 0
        )


class RandomClipSampler(Sampler):
    def __init__(self, video_clips, max_clips_per_video):
        self.video_clips = video_clips
        self.max_clips_per_video = max_clips_per_video

    def __iter__(self):
        idxs = []
        s = 0
        # select at most max_clips_per_video for each video, randomly
        for c in self.video_clips.frame_list:
            length = len(c)
            size = min(length, self.max_clips_per_video)
            sampled = torch.randperm(length)[:size] + s
            s += length
            idxs.append(sampled)
        idxs_ = torch.cat(idxs)
        # shuffle all clips randomly
        perm = torch.randperm(len(idxs_))
        return iter(idxs_[perm].tolist())

    def __len__(self) -> int:
        return sum(min(len(c), self.max_clips_per_video) for c in self.video_clips.frame_list)


class UniformClipSampler_val(Sampler):
    def __init__(self, video_clips, num_clips_per_video, shuffle=True):
        self.video_clips = video_clips
        self.num_clips_per_video = num_clips_per_video
        self.shuffle = shuffle

    def __iter__(self):
        idxs = []
        s = 0
        # select num_clips_per_video for each video, uniformly spaced
        for c in self.video_clips.frame_list:
            length = len(c)
            if length == 0:
                # corner case where video decoding fails
                continue

            sampled = (
                torch.linspace(s, s + length - 1, steps=self.num_clips_per_video)
                    .floor()
                    .to(torch.int64)
            )

            if length < self.num_clips_per_video:
                sampled = sampled = (
                    torch.linspace(s, s + length - 1, steps=length)
                        .floor()
                        .to(torch.int64)
                )
            # print(length, sampled)

            s += length
            idxs.append(sampled)
        idxs_ = torch.cat(idxs)
        # print(idxs_)
        # exit()
        # shuffle all clips randomly
        if self.shuffle:
            perm = torch.randperm(len(idxs_))
            idxs = idxs_[perm]
        else:
            idxs = idxs_
        return iter(idxs.tolist())

    def __len__(self) -> int:
        return sum(
            self.num_clips_per_video for c in self.video_clips.frame_list if len(c) > 0
        )
