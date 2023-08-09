# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmcv import Config


from nat import *
from dinat import *
from dinats import *
from extras import get_gflops, get_mparams

from mmseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', default='configs/nat/dense_nat_base.py', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    inp = torch.randn(input_shape)
    if torch.cuda.is_available():
        model.cuda()
        inp = inp.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops = get_gflops(model, inp)
    params = get_mparams(model)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2:.3f} * 1e9\nParams: {3:.3f} * 1e6\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
