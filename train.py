import os

import torch as t
from torch.utils.data import DataLoader
import torch
from dataset.MP_dataset import MP_dataset
from sampler import UniformClipSampler, RandomClipSampler
from utils import AverageMeter, accuracy, my_parse_args

import slowfast.utils.checkpoint as cu
from slowfast.models import build_model
from slowfast.utils.parser import load_config


def train_FG(data_loader, obj_model, optimizer, device):
    all_loss = AverageMeter()
    all_hit1 = AverageMeter()
    all_len = len(data_loader)
    for i, (x, v_ids, v_idxs, label, is_last_clip) in enumerate(data_loader):
        optimizer.zero_grad()
        #########  CUDA
        # print(len(x))
        # print(x[0].shape)

        x = [y.to(device) for y in x]
        # count_ops(obj_model, x[0])
        y_pred = obj_model(x)
        loss = torch.nn.CrossEntropyLoss()(y_pred, label.to(device))
        # loss = torch.nn.CrossEntropyLoss()(y_pred, label)
        all_loss.update(loss.item(), len(x))
        [hit1] = accuracy(y_pred.detach().cpu(), label, topk=(1,))
        all_hit1.update(hit1.item(), len(x))
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('{}/{} current loss is {} hit1 is {} avg train loss is {} hit1 is {}'.format(
                i, all_len, loss.item(), hit1.item(), all_loss.avg, all_hit1.avg))

    print('{}/{} train loss is {} hit1 is {}'.format(i, all_len, all_loss.avg, all_hit1.avg))
    return obj_model


def val_FG(data_loader, obj_model, device):
    all_loss = AverageMeter()
    all_hit1 = AverageMeter()
    all_len = len(data_loader)
    for i, (x, v_ids, v_idxs, label, is_last_clip) in enumerate(data_loader):
        #########  CUDA
        x = [y.to(device) for y in x]
        y_pred = obj_model(x)
        # loss = torch.nn.CrossEntropyLoss()(y_pred, label)
        loss = torch.nn.CrossEntropyLoss()(y_pred, label.to(device))
        all_loss.update(loss.item(), len(x))
        [hit1] = accuracy(y_pred.detach().cpu(), label, topk=(1,))
        all_hit1.update(hit1.item(), len(x))

        if i % 100 == 0:
            print('{}/{} avg val loss is {} hit1 is {}'.format(i, all_len, all_loss.avg, all_hit1.avg))
    print('{}/{} avg val loss is {} hit1 is {}'.format(i, all_len, all_loss.avg, all_hit1.avg))
    return all_loss.avg, all_hit1.avg


def _build_model(args):
    cfg = load_config(args)
    cfg.NUM_GPUS = 1
    model = build_model(cfg, 0)
    return model, cfg


def init_model(model_config, weight_path):
    args = my_parse_args(model_config)
    model, cfg = _build_model(args)
    convert_from_caffe2 = True if '.pkl' in weight_path else False
    cu.load_checkpoint(weight_path, model, data_parallel=False, convert_from_caffe2=convert_from_caffe2)
    tmp_config = model_config.split('/')[-1]
    print(tmp_config)
    return model, cfg


def start(num_frame, sampling_rate, model_config, weight_path,
          batch_size=2, epoch=1, data_path=''):
    model, cfg = init_model(model_config, weight_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = None
    a = MP_dataset('train', num_frame * sampling_rate, 1, size=size, every_n_skip=sampling_rate, cfg=cfg,
                   data_path=data_path)
    a_val = MP_dataset('val', num_frame * sampling_rate, 1, size=size, every_n_skip=sampling_rate, cfg=cfg,
                        data_path=data_path)

    batch_sampler_train = torch.utils.data.BatchSampler(RandomClipSampler(a.video_list, 2), batch_size, drop_last=False)
    batch_sampler_test = torch.utils.data.BatchSampler(UniformClipSampler(a_val.video_list, 10), batch_size,
                                                       drop_last=False)
    dl = DataLoader(a, num_workers=4, batch_sampler=batch_sampler_train)
    dl_val = DataLoader(a_val, num_workers=4, batch_sampler=batch_sampler_test)


    #########  CUDA
    model.to(device)

    lr = 0.001
    # optimizer = optim.construct_optimizer(model, cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.9)

    (best_loss, best_hit1, best_epoch) = (100, 0, 0)

    print(os.getcwd())
    for i in range(epoch):
        model.train()
        model = train_FG(dl, model, optimizer, device)
        print(i, '-' * 20)
        with t.no_grad():
            model.eval()
            new_loss, new_hit1 = val_FG(dl_val, model, device)
        if new_hit1 > best_hit1 or new_loss < best_loss:
            best_hit1 = new_hit1
            best_loss = new_loss
            best_epoch = i
            torch.save(model.state_dict(), './trained_weights/{}_{:.4f}_{:.4f}.pt'.format(i, new_hit1, new_loss))
        scheduler2.step()

        print(i, '-' * 20)


if __name__ == '__main__':
    """
    start function parametres

    Args:
        num_frame: number of frames
        sampling_rate: skip every n frames/temporal sub-sampling rate
        model_config: path to model config
        weight_path: path to pretrained weights
        save_path: path for saving model weights during training.
                   Please note that only models with hit1 or loss better than previous best model will be saved.
                   If every model needs to be saved codes need to be changed.
        batch_size: default=10
        epoch: default=15
        data_path: path to dataset images only. defualt: '/home/feiyan/dataset/VideoMPSplit/'

    Returns:
        None
    """

    # start(32, 2, "slowfast/configs/Kinetics/c2/SLOWFAST_4x16_R50.yaml",
    #       '/home/feiyan/Github/Code/UCD-MP/trained_weights/SLOWFAST_4x16_R50.pkl',
    #       'SLOWFAST_4x16_R50_newdtloader2_aug2_val2')
    #
    '''
    start(32, 2, "slowfast/configs/Kinetics/SLOWFAST_NLN_4x16_R50.yaml", 
          '/home/feiyan/Github/Code/UCD-MP/trained_weights/SLOWFAST_4x16_R50.pkl',
          'SLOWFAST_NLN_4x16_R50_newdtloader2_aug2_val2')
    '''
    '''
    start(8, 8, "slowfast/configs/Kinetics/c2/I3D_8x8_R50.yaml",
          '/home/feiyan/Github/Code/UCD-MP/trained_weights/I3D_8x8_R50.pkl',
          'I3D_8x8_R50_newdtloader2_aug2_val')
    '''
    ################# Local
    # start(8, 8, model_config="/home/ashish/PycharmProjects/train_test_MP/configs/C2D_8x8_R50_local.yaml",
    #       weight_path='/home/ashish/PycharmProjects/train_test_MP/pretrained/C2D_8x8_R50.pyth', batch_size=2, epoch=1,
    #       data_path='/home/ashish/Downloads/FullMP/')

    # ################# Slowfast NLN
    start(32, 2, model_config="/home/ashish/Projects/train_test_MP/configs/pytorchvideo/SLOWFAST_NLN_4x16_R50.yaml",
          weight_path='/home/ashish/Projects/pretrained/SLOWFAST_4x16_R50.pyth', batch_size=5, epoch=10,
          data_path='/home/ashish/Results/Datasets/HPE2/FullMP/')

    # ################# Slowfast
    # start(32, 2, model_config="/home/ashish/Projects/train_test_MP/configs/pytorchvideo/SLOWFAST_4x16_R50.yaml",
    #       weight_path='/home/ashish/Projects/pretrained/SLOWFAST_4x16_R50.pyth', batch_size=5, epoch=10,
    #       data_path='/home/ashish/Results/Datasets/HPE2/FullMP/')

    ################# C2D
    # start(8, 8, model_config="/home/ashish/Projects/train_test_MP/configs/pytorchvideo/C2D_8x8_R50.yaml",
    #       weight_path='/home/ashish/Projects/pretrained/C2D_8x8_R50.pyth', batch_size=10, epoch=10,
    #       data_path='/home/ashish/Results/Datasets/HPE2/FullMP/')

    ################# I3D
    # start(8, 8, model_config="/home/ashish/Projects/train_test_MP/configs/pytorchvideo/I3D_8x8_R50.yaml",
    #       weight_path='/home/ashish/Projects/pretrained/I3D_8x8_R50.pyth', batch_size=5, epoch=10,
    #       data_path='/home/ashish/Results/Datasets/HPE2/FullMP/')

    # start(16, 5, model_config="/home/ashish/Projects/train_test_MP/configs/pytorchvideo/X3D_M.yaml",
    #       weight_path='/home/ashish/Projects/pretrained/X3D_M.pyth', batch_size=5, epoch=10,
    #       data_path='/home/ashish/Results/Datasets/HPE2/FullMP/')

    '''
    start(16, 4, "slowfast/configs/Kinetics/pytorchvideo/R2PLUS1D_16x4_R50.yaml",
          '/home/feiyan/Github/Code/UCD-MP/trained_weights/R2PLUS1D_16x4_R50.pyth',
          'R2PLUS1D_16x4_R50_newdtloader2_aug2_val', batch_size=6)
    '''
    '''
    start(16, 5, "slowfast/configs/Kinetics/X3D_L.yaml",
          '/home/feiyan/Github/Code/UCD-MP/trained_weights/x3d_l.pyth',
          'X3D_L_newdtloader2_aug2_val', batch_size=3)
    '''
