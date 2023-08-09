import os
import time
import datetime

import torch

# from networks import AESwin as AE
from networks import AEViT as AE
from utils import misc, train_one_epoch, evaluate, PolyLR
from my_dataset import Dataset, DataTransform


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch AESwin training")

    parser.add_argument("--data-path", default="/home/long/dataset/planning224_1_seg_generalization", help="dataset root")
    parser.add_argument("--img_channels", default=1, type=int)
    parser.add_argument("--out_channels", default=1, type=int)
    parser.add_argument("--resolution", default=224, type=int)
    parser.add_argument("--embedding_dim", default=24, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=3, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--lr_power", type=int, default=0.9,
                        help="poly's power")
    parser.add_argument("--min_lr", type=int, default=1e-6)
    parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume',
                        # default="/home/long/fine-tune/PlanningNet/2nd_aeswin_320k_path_patch_C1/model_0.pth",
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # 创建模型 打印参数
    model = AE(img_channels=args.img_channels,
               out_channels=args.out_channels,
               img_resolution=args.resolution,
               dim=args.embedding_dim)
    model.to(device)
    img_in = torch.empty([batch_size, args.img_channels, args.resolution, args.resolution], device=device)
    misc.print_module_summary(model, img_in)

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = Dataset(args.data_path, transforms=DataTransform(), subset='train')
    val_dataset = Dataset(args.data_path, transforms=DataTransform(), subset='val')

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    params_to_optimize = [
        {"params": [p for p in model.parameters() if p.requires_grad]}
    ]

    # optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=0, betas=(0, 0.99), eps=1e-8)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    if args.lr_policy == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    else:
        lr_scheduler = PolyLR(optimizer, len(train_loader)*args.epochs, power=args.lr_power, min_lr=args.min_lr)

    # import matplotlib.pyplot as plt
    # lr_list = []
    # for _ in range(args.epochs):
    #     for _ in range(len(train_loader)):
    #         lr_scheduler.step()
    #         lr = optimizer.param_groups[0]["lr"]
    #         lr_list.append(lr)
    # plt.plot(range(len(lr_list)), lr_list)
    # plt.show()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        loss = evaluate(model, val_loader, device=device)
        val_info = 'loss:' + str(loss)
        print(val_info)
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        torch.save(save_file, "./save_weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


if __name__ == '__main__':
    main()
