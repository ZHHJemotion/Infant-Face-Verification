import os
import time
import math
import cv2
import argparse
import numpy as np
import multiprocessing

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

from tqdm import tqdm
from scripts.losses import ContrastiveLoss
from scripts.utils import accuracy, init_weights
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from models.deepid2_inception import DeepID2_Inception
from data.data_loader import SiameseNetworkDataLoader


def train(args):
    weight_dir = os.path.join(args.log_root, 'Inception/weights')
    log_dir = os.path.join(args.log_root, 'logs', 'DS-Net-{}'.format(time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                                   time.localtime())))

    data_dir = os.path.join(args.data_root, args.dataset)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 1. Setup DataLoader
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> 0. Setting up DataLoader...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((152.0643, 111.4953, 97.5015), (52.6354, 45.5230, 41.9734))
    ])

    train_loader = SiameseNetworkDataLoader(data_dir, split='train', num_pair=args.train_pair,
                                            img_size=(args.img_row, args.img_col), transform=transform)

    num_classes = train_loader.num_classes

    valid_loader = SiameseNetworkDataLoader(data_dir, split="val", num_pair=args.val_pair,
                                            img_size=(args.img_row, args.img_col), transform=transform)

    tra_loader = data.DataLoader(train_loader, batch_size=args.batch_size,
                                 num_workers=int(multiprocessing.cpu_count() / 2),
                                 shuffle=True, collate_fn=train_loader.collate_fn)
    val_loader = data.DataLoader(valid_loader, batch_size=args.batch_size,
                                 num_workers=int(multiprocessing.cpu_count() / 2),
                                 shuffle=True, collate_fn=train_loader.collate_fn)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 2. Setup Model
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> 1. Setting up Model...")
    model = DeepID2_Inception()
    # model = DataParallelModel(model, device_ids=[0, 1, 2]).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    # 2.1 Setup Optimizer
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # Check if model has custom optimizer
    if hasattr(model.module, 'optimizer'):
        print('> Using custom optimizer')
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.90,
                                    weight_decay=5e-4, nesterov=True)
        # torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1e3, norm_type=float('inf'))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # 2.2 Setup Loss
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    loss_contrastive = ContrastiveLoss()
    loss_CE = torch.nn.CrossEntropyLoss()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 3. Resume Model
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> 2. Model state init or resume...")
    args.start_epoch = 0
    beat_map = -100
    if args.resume is not None:
        full_path = os.path.join(weight_dir, args.resume)
        if os.path.isfile(full_path):
            print("> Loading model and optimizer from checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(full_path)

            args.start_epoch = checkpoint['epoch']
            # beat_map = checkpoint['beat_map']
            model.load_state_dict(checkpoint['model_state'])  # weights
            optimizer.load_state_dict(checkpoint['optimizer_state'])  # gradient state
            del checkpoint

            print("> Loaded checkpoint '{}' (epoch {})".format(args.resume, args.start_epoch))

        else:
            print("> No checkpoint found at '{}'".format(full_path))
            raise Exception("> No checkpoint found at '{}'".format(full_path))
    else:
        # init_weights(model, pi=0.01,
        #              pre_trained="/home/pingguo/PycharmProject/dl_project/Weights/DS-Net/mobilenetv2.pth.tar")
        init_weights(model=model, pre_trained=None)

        if args.pre_trained is not None:
            print("> Loading weights from pre-trained model '{}'".format(args.pre_trained))
            full_path = os.path.join(weight_dir, args.pre_trained)

            pre_weight = torch.load(full_path)

            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pre_weight.items() if k in model_dict}

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            del pre_weight
            # del model_dict
            model_dict = None
            del pretrained_dict

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 4. Train Model
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 4.0. Setup tensor-board for visualization
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    writer = None
    if args.tensor_board:
        writer = SummaryWriter(log_dir=log_dir, comment="Face_Verification")
        dummy_input_1 = Variable(torch.rand(1, 3, args.img_row, args.img_col).cuda(), requires_grad=True)
        dummy_input_2 = Variable(torch.rand(1, 3, args.img_row, args.img_col).cuda(), requires_grad=True)
        # writer.add_graph(model, (dummy_input_1, dummy_input_2))

    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> 3. Model Training start...")
    num_batches = int(math.ceil(tra_loader.dataset.num_pair / float(tra_loader.batch_size)))

    for epoch in np.arange(args.start_epoch, args.num_epochs):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 4.1 Mini-Batch Training
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        model.train()
        pbar = tqdm(np.arange(num_batches))
        for train_i, (labels, images_1, images_2) in enumerate(tra_loader):  # One mini-Batch data, One iteration
            full_iter = (epoch * num_batches) + train_i + 1

            pbar.update(1)
            pbar.set_description("> Epoch [%d/%d]" % (epoch + 1, args.num_epochs))

            images_1 = Variable(images_1.cuda(), requires_grad=True)  # Image feed into the deep neural network
            images_2 = Variable(images_2.cuda(), requires_grad=True)
            labels = Variable(labels.cuda(), requires_grad=False)

            optimizer.zero_grad()
            feats_1, feats_2, aux_diff_preds, diff_preds = model(images_1, images_2)  # Here we have 3 output

            # !!!!!! Please Loss define !!!!!!
            targets = labels.type(torch.cuda.FloatTensor)
            loss_1 = loss_contrastive(feats_1, feats_2, targets)

            loss_aux = loss_CE(aux_diff_preds, labels[:, 0])

            loss_2 = loss_CE(diff_preds, labels[:, 0])
            losses = loss_1 + loss_2 + 0.3 * loss_aux
            losses.backward()  # back-propagation

            torch.nn.utils.clip_grad_norm(model.parameters(), 1e3)
            optimizer.step()  # parameter update based on the current gradient

            """
            if full_iter % 3000 == 0:
                state = model.state_dict()

                save_dir = os.path.join(weight_dir, "dsnet_model.pkl")
                torch.save(state, save_dir)
            """
            pbar.set_postfix(Losses=losses.data[0])

            # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
            # 4.1.1 Verbose training process
            # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
            if (train_i + 1) % args.verbose_interval == 0:
                # ---------------------------------------- #
                # 1. Training Losses
                # ---------------------------------------- #
                loss_log = "Epoch [%d/%d], Iter: %d Loss1: \t %.4f" % (epoch + 1, args.num_epochs,
                                                                       train_i + 1, losses.data[0])

                # ---------------------------------------- #
                # 2. Training Metrics
                # ---------------------------------------- #
                # convert reg_pred from tx, ty, tw, th to xmin, ymin, xmax, ymax
                cls_preds = F.softmax(diff_preds, dim=1)
                prec = accuracy(cls_preds, labels)
                pbar.set_postfix(Accuracy=prec.data[0])

                metric_log = "Epoch [%d/%d], Iter: %d, Acc: \t %.3f" % (epoch + 1, args.num_epochs,
                                                                        train_i + 1, prec.data[0])

                logs = loss_log + metric_log
                if args.tensor_board:
                    writer.add_scalar('Training/Loss', losses.data[0], full_iter)
                    writer.add_scalar('Training/Accuracy', prec.data[0], full_iter)

                    writer.add_text('Training/Text', logs, full_iter)

                    for name, param in model.named_parameters():
                        param_value = param.clone().cpu().data.numpy()
                        writer.add_histogram(name, param_value, full_iter)

        state = {"epoch": epoch + 1,
                 "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict()}

        save_dir = os.path.join(weight_dir, "faceVerification_Inception_model.pkl")
        torch.save(state, save_dir)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 4.2 Mini-Batch Validation
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        model.eval()

        val_loss = 0.0
        acc_val = 0.0
        vali_count = 0
        for i_val, (labels_val, images_1_val, images_2_val) in enumerate(val_loader):
            vali_count += 1

            images_1_val = Variable(images_1_val.cuda(), volatile=True)
            images_2_val = Variable(images_2_val.cuda(), volatile=True)
            labels_val = Variable(labels_val.cuda(), volatile=True)

            feats_1, feats_2, aux_diff_preds_val, preds_val = model(images_1_val, images_2_val)  # Here we have 4 output for 4 loss

            # !!!!!! Please Loss define !!!!!!
            targets_val = labels_val.type(torch.cuda.FloatTensor)
            loss_1 = loss_contrastive(feats_1, feats_2, targets_val)
            loss_aux = loss_CE(aux_diff_preds_val, labels_val[:, 0])
            loss_2 = loss_CE(preds_val, labels_val[:, 0])
            val_losses = loss_1 + loss_2 + 0.3 * loss_aux
            val_loss += val_losses.data[0]

            # !!!!! Here calculate Metrics !!!!!
            # accumulating the confusion matrix and ious
            preds_val = F.softmax(preds_val, dim=1)
            prec_val = accuracy(preds_val, labels_val)
            acc_val += prec_val.data[0]

            # labels_val = labels_val.type(torch.cuda.LongTensor)

        # ---------------------------------------- #
        # 1. Validation Losses
        # ---------------------------------------- #
        val_loss /= vali_count
        acc_val /= vali_count

        loss_log = "Epoch [%d/%d], Loss: \t %.4f" % (epoch + 1, args.num_epochs, val_loss)

        # ---------------------------------------- #
        # 2. Validation Metrics
        # ---------------------------------------- #
        metric_log = "Epoch [%d/%d], Acc: \t %.3f" % (epoch + 1, args.num_epochs, acc_val)

        logs = loss_log + metric_log

        if args.tensor_board:
            writer.add_scalar('Validation/Loss', val_loss, epoch)
            writer.add_scalars('Validation/Accuracy', acc_val, epoch)

            writer.add_text('Validation/Text', logs, epoch)

            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 4.3 End of one Epoch
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # !!!!! Here choose suitable Metric for the best model selection !!!!!

        if acc_val >= beat_map:
            beat_map = acc_val
            state = {"epoch": epoch + 1,
                     "beat_map": beat_map,
                     "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict()}

            save_dir = os.path.join(weight_dir, "faceVerification_inception_best_model.pkl")
            torch.save(state, save_dir)

        # Note that step should be called after validate()
        scheduler.step()

        pbar.close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 4.4 End of Training process
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    if args.tensor_board:
        # export scalar data to JSON for external processing
        # writer.export_scalars_to_json("{}/all_scalars.json".format(log_dir))
        writer.close()
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")
    print("> Training Done!!!")
    print("> # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #")


if __name__ == "__main__":
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 0. Hyper-params
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    parser = argparse.ArgumentParser(description='Hyper-params')

    parser.add_argument('--dataset', nargs='?', type=str, default='faceDatasetCV/face_31&11',
                        help='Dataset to use | faceDataset by  default')
    parser.add_argument('--data_root', nargs='?', type=str, default='/home/pingguo/ril-server/PycharmProject/database/',
                        help='Dataset to use | /home/pingguo/ril-server by  default')

    parser.add_argument('--train_pair', nargs='?', type=int, default=256000,
                        help='Number of train image pairs | 256000 by  default')
    parser.add_argument('--val_pair', nargs='?', type=int, default=64000,
                        help='Number of validation image pairs | 64000 by  default')

    parser.add_argument('--img_row', nargs='?', type=int, default=128,
                        help='Height of the input image | 128 by  default')
    parser.add_argument('--img_col', nargs='?', type=int, default=128,
                        help='Width of the input image | 128 by  default')

    parser.add_argument('--num_epochs', nargs='?', type=int, default=60,
                        help='# of the epochs used for training process | 120 by  default')
    parser.add_argument('--verbose_interval', nargs='?', default=100,  # 60
                        help='The interval for training result verbose | 60 by  default')
    parser.add_argument('--batch_size', nargs='?', default=64,  # 32
                        help='Batch size | 32 by  default')
    parser.add_argument('--learning_rate', nargs='?', type=float, default=5e-4,
                        help='Learning rate | 2.5e-3 by  default')

    parser.add_argument('--resume', nargs='?', type=str, default=None,  # 'faceVerification_model.pkl',
                        help='Path to previous saved model to restart from | None by  default')
    parser.add_argument('--pre_trained', nargs='?', type=str, default=None,
                        help='Path to pre-trained  model to init from | None by  default')

    parser.add_argument('--tensor_board', nargs='?', type=bool, default=True,
                        help='Show visualization(s) through tensor-board | True by  default')
    parser.add_argument('--log_root', nargs='?', type=str, default='/home/pingguo/PycharmProject/Weights/',
                        help='Dataset to use | /home/pingguo/PycharmProject/Weights/faceVerification by  default')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # 1. Train the Deep Model
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train_args = parser.parse_args()
    train(train_args)
