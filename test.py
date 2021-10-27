import torch as t
from torch.utils.data import DataLoader
import torch
from dataset.MP_dataset import MP_dataset
from utils import AverageMeter, accuracy, my_parse_args
from sampler import UniformClipSampler_val

from slowfast.models import build_model
from slowfast.utils.parser import load_config


def test_FG(data_loader, obj_model, resize_config, device):
    (all_pred, all_islast, all_label, all_vid) = ([], [], [], [])
    all_vid_count = 0
    all_hit1 = AverageMeter()
    all_len = len(data_loader)
    all_pred_save = {}
    for i, (x, v_ids, v_idxs, label, is_last_clip) in enumerate(data_loader):

        print(i)
        x = [y.view(-1, 3, tmp_config[0], tmp_config[1], tmp_config[2]) for tmp_config, y in zip(resize_config, x)]
        print(x[0].shape)
        v_ids = [tmp.split('/')[-1] for tmp in v_ids]
        print(v_ids)
        #####################
        x = [y.to(device) for y in x]
        label = label.to(device)

        y_pred = obj_model(x)
        y_pred = y_pred.view(-1, 3, 4)
        y_pred = torch.mean(y_pred, dim=1)
        # loss = torch.nn.CrossEntropyLoss()(y_pred, label.cuda())
        if len(all_pred) == 0:
            all_pred = y_pred.detach()
            all_label = label
            all_islast = is_last_clip
            all_vid = v_ids
        else:
            all_pred = torch.cat([all_pred, y_pred.detach()], dim=0)
            all_islast = torch.cat([all_islast, is_last_clip], dim=0)
            all_label = torch.cat([all_label, label], dim=0)
            all_vid = all_vid + v_ids
        # print(is_last_clip, v_ids, all_pred.shape, all_islast.shape, all_label.shape)
        # print(label)

        start_idx = [j + 1 for j, islast in enumerate(all_islast) if islast]
        start_idx = [0] + start_idx
        for j in range(len(start_idx) - 1):
            # avg_pred8 = all_pred_save_8[all_vid_count]
            all_vid_count += 1
            (start_i, end_i) = (start_idx[j], start_idx[j + 1])
            tmp_vid = all_vid[start_i:end_i]
            avg_pred = torch.mean(all_pred[start_i:end_i], dim=0)
            # all_pred_save[tmp_vid[0]] = avg_pred.detach().cpu().numpy()
            [hit1] = accuracy(avg_pred.unsqueeze(0), all_label[start_i:start_i + 1], topk=(1,))
            all_hit1.update(hit1.item())
        all_pred = all_pred[start_idx[-1]:]
        all_islast = all_islast[start_idx[-1]:]
        all_label = all_label[start_idx[-1]:]
        all_vid = all_vid[start_idx[-1]:]
        if i % 10 == 0:
            print('vid{} {}/{}, avg hit1 is {}'.format(all_vid_count, i, all_len, all_hit1.avg))
    print('vid{} {}/{}, avg hit1 is {}'.format(all_vid_count, i, all_len, all_hit1.avg))
    return all_hit1.avg


def _build_model(args):
    cfg = load_config(args)
    cfg.NUM_GPUS = 1
    model = build_model(cfg, 0)
    return model, cfg


def init_model(model_config, weight_path):
    args = my_parse_args(model_config)
    model, cfg = _build_model(args)
    model.load_state_dict(torch.load(weight_path))
    return model, cfg


def start(num_frame, sampling_rate, model_config, weight_path,
          batch_size=2, data_path=''):
    model, cfg = init_model(model_config, weight_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = None
    a_test = MP_dataset('test', num_frame * sampling_rate, 1, size=size, every_n_skip=sampling_rate, cfg=cfg,
                        data_path=data_path)

    batch_sampler_test = torch.utils.data.BatchSampler(UniformClipSampler_val(a_test.video_list, 10, shuffle=False),
                                                       batch_size, drop_last=False)
    dl_test = DataLoader(a_test, num_workers=4, batch_sampler=batch_sampler_test)

    model.to(device)
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        resize_config = [(num_frame, cfg.DATA.TEST_CROP_SIZE, cfg.DATA.TEST_CROP_SIZE)]
    else:
        resize_config = [(int(num_frame / 8), cfg.DATA.TEST_CROP_SIZE, cfg.DATA.TEST_CROP_SIZE),
                         (num_frame, cfg.DATA.TEST_CROP_SIZE, cfg.DATA.TEST_CROP_SIZE)]

    with t.no_grad():
        model.eval()
        new_hit1 = test_FG(dl_test, model, resize_config, device)


if __name__ == '__main__':
    """
    start function parametres for 10 views and 3 crops testing

    Args:
        num_frame: number of frames
        sampling_rate: skip every n frames/temporal sub-sampling rate
        model_config: path to model config
        weight_path: path to pretrained weights
        batch_size: default=10
        data_path: path to dataset images only. defualt: '/home/feiyan/dataset/VideoMPSplit/'

    Returns:
        None
    """

    # start(8, 8, model_config="/home/ashish/PycharmProjects/train_test_MP/configs/C2D_8x8_R50_local.yaml",
    #       weight_path='/home/ashish/PycharmProjects/train_test_MP/trained_weights/0_40.0000_1.3746.pt',
    #       data_path='/home/ashish/Downloads/FullMP/')

    start(32, 2, model_config="/home/ashish/Projects/train_test_MP/configs/pytorchvideo/SLOWFAST_NLN_4x16_R50.yaml",
          weight_path='/home/ashish/Projects/train_test_MP/trained_weights/9_94.0000_0.8007.pt', batch_size=10,
          data_path='/home/ashish/Results/Datasets/HPE2/FullMP/')

    # start(32, 2, model_config="/home/ashish/Projects/train_test_MP/configs/pytorchvideo/SLOWFAST_4x16_R50.yaml",
    #       weight_path='/home/ashish/Projects/train_test_MP/trained_weights/9_94.0000_0.8007.pt', batch_size=10,
    #       data_path='/home/ashish/Results/Datasets/HPE2/FullMP/')

    # start(8, 8, model_config="/home/ashish/Projects/train_test_MP/configs/pytorchvideo/I3D_8x8_R50.yaml",
    #       weight_path='/home/ashish/Projects/train_test_MP/trained_weights/9_92.0000_0.8495.pt', batch_size=10,
    #       data_path='/home/ashish/Results/Datasets/HPE2/FullMP/')

    # start(16, 5, model_config="/home/ashish/Projects/train_test_MP/configs/pytorchvideo/X3D_M.yaml",
    #       weight_path='/home/ashish/Projects/train_test_MP/trained_weights/9_92.0000_0.8495.pt', batch_size=10,
    #       data_path='/home/ashish/Results/Datasets/HPE2/FullMP/')

    # start(8, 8, model_config="/home/ashish/Projects/train_test_MP/configs/pytorchvideo/C2D_8x8_R50.yaml",
    #       weight_path='/home/ashish/Projects/train_test_MP/C2D_8x8_R50.pyth', batch_size=10,
    #       data_path='/home/ashish/Results/Datasets/HPE2/FullMP/')

    # start(32, 2, "slowfast/configs/Kinetics/c2/SLOWFAST_4x16_R50.yaml",
    #      '../trained_weights/SLOWFAST_4x16_R50_newdtloader2_aug2_val/10_88.3750_0.8713.pt')
    # start(32, 2, "slowfast/configs/Kinetics/SLOWFAST_NLN_4x16_R50.yaml",
    #      '../trained_weights/SLOWFAST_NLN_4x16_R50_newdtloader2_aug2_val/4_85.3750_0.9075.pt')
    # start(8, 8, "slowfast/configs/Kinetics/c2/I3D_8x8_R50.yaml",
    #      '../trained_weights/I3D_8x8_R50_newdtloader2_aug2_val/10_82.3125_0.9360.pt')
    # start(8, 8, "slowfast/configs/Kinetics/pytorchvideo/C2D_8x8_R50.yaml",
    #      '../trained_weights/C2D_8x8_R50_newdtloader2_aug2_val/7_76.3750_0.9839.pt')
    # start(16, 4, "slowfast/configs/Kinetics/pytorchvideo/R2PLUS1D_16x4_R50.yaml",
    #      '../trained_weights/R2PLUS1D_16x4_R50_newdtloader2_aug2_val/4_83.4375_0.9224.pt')
    # start(16, 5, "slowfast/configs/Kinetics/X3D_L.yaml",
    #       '../trained_weights/X3D_L_newdtloader2_aug2_val/11_91.8125_0.8421.pt')
