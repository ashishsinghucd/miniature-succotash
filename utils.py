import torch
import argparse


# other util
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def to_flowinput(x, m, resize=None):
    all_flowx = []
    x = x.float().cuda()
    for i in range(x.shape[1] - 1):
        tmp = torch.stack([x[:, i, :, :, :], x[:, i + 1, :, :, :]])
        tmp = tmp.permute(1, 4, 0, 2, 3)
        tmp = m(tmp)
        if resize:
            tmp = torch.nn.functional.interpolate(tmp, size=resize, mode='bilinear')
        all_flowx.append(tmp.detach())
    all_flowx = torch.stack(all_flowx)
    all_flowx = all_flowx.permute(1, 2, 0, 3, 4)
    # tmp = torch.sqrt(all_flowx[:, 0:1, :, :, :]**2 + all_flowx[:, 1:2, :, :, :]**2)
    tmp = all_flowx[:, 0:1, :, :, :] * all_flowx[:, 1:2, :, :, :]
    all_flowx = torch.cat([all_flowx, tmp], dim=1)
    return all_flowx
    print(torch.max(x), torch.min(x), all_flowx.shape)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def my_parse_args(model_path):
    import sys
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )

    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default=model_path,
        type=str,
    )

    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--NUM_GPUS",
        default=1,
        type=int
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()
