import argparse
import os
import torch
import random
import datetime
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model_new import ARCNN, VRCNN, ARCNN_Auto, VDSR, U_Net, SSIU_Net
from model import PtRestoration, SE_VRCNN
# from DenseNet import DenseNet121
from dataset import *
from utils import AverageMeter, cal_psnr, mse2psnr
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("./events/U-Net_r02_to_r06")
cudnn.benchmark = True
# os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = lr * (0.2 ** (epoch // 50))
    # optimizer.param_groups[0]['lr'] = lr
    # optimizer.param_groups[1]['lr'] = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    bits = 'R05_'
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='U_Net', help='ARCNN or ... or SSIU_Net')
    parser.add_argument('--pth_path', type=str, default='./pths/2023/U_Net')
    parser.add_argument('--log_path', type=str, default='./logs/2023/U_Net')
    # parser.add_argument('--attrQuantizationParam', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=121)
    parser.add_argument('--lr', type=float, default=0.2e-4)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--use_augmentation', action='store_true')
    parser.add_argument('--use_fast_loader', action='store_true')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--bit_rate_point', type=str, default='R01_R05_U_Net')
    parser.add_argument('--data_path', type=str, default='./dataset_creat/LT/h5_WPCSD/trainFile_r05.txt')
    parser.add_argument('--validData_path', type=str, default='./dataset_creat/LT/h5_WPCSD_test/testFile_r05.txt')
    parser.add_argument('--checkpoint', type=str, default='./pths/2021-12-18/r02_manual_people_y_UNet/model_40.pth')
    parser.add_argument('--notes', type=str, default='Lift_r01_to_r05_U_Net')
    opt = parser.parse_args()
    daytime = datetime.datetime.now().strftime('%Y-%m-%d')   # year,month,day

    model_path_Y = opt.pth_path + '/Y/' + daytime + '/' + bits + str(opt.arch)
    model_path_U = opt.pth_path + '/U/' + daytime + '/' + bits + str(opt.arch)
    model_path_V = opt.pth_path + '/V/' + daytime + '/' + bits + str(opt.arch)
    if not os.path.exists(model_path_Y):
        os.makedirs(model_path_Y)
    if not os.path.exists(model_path_U):
        os.makedirs(model_path_U)
    if not os.path.exists(model_path_V):
        os.makedirs(model_path_V)
    logs_path_Y = opt.log_path + '/' + daytime + '/' + bits + str(opt.arch) + '/Y'
    logs_path_U = opt.log_path + '/' + daytime + '/' + bits + str(opt.arch) + '/U'
    logs_path_V = opt.log_path + '/' + daytime + '/' + bits + str(opt.arch) + '/V'
    if not os.path.exists(logs_path_Y):
        os.makedirs(logs_path_Y)
    if not os.path.exists(logs_path_U):
        os.makedirs(logs_path_U)
    if not os.path.exists(logs_path_V):
        os.makedirs(logs_path_V)
    torch.manual_seed(opt.seed)
    model1 = U_Net().to(device)
    model2 = U_Net().to(device)
    model3 = U_Net().to(device)
    # if opt.arch == 'ARCNN':
    #     model = ARCNN()
    # elif opt.arch == 'VRCNN':
    #     model = VRCNN()
    # elif opt.arch == 'ARCNN_Auto':
    #     model = ARCNN_Auto()
    # elif opt.arch == 'PtRestoration':
    #     model = PtRestoration()
    # elif opt.arch == 'SE_VRCNN':
    #     model = SE_VRCNN()
    # elif opt.arch == 'VDSR':
    #     model = VDSR()
    # elif  opt.arch == 'SSIU_Net':
    #     model = U_Net()
    # txt to store the loss, log,,,,
    txt_loss_mse = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_loss_Y.txt'
    txt_mse_path_Y = os.path.join(logs_path_Y, txt_loss_mse)
    txt_loss_psnr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_psnr_Y.txt'
    txt_psnr_path_Y = os.path.join(logs_path_Y, txt_loss_psnr)
    txtValid_name_loss = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_Valid_loss_Y.txt'
    txt_lossValid_path_Y = os.path.join(logs_path_Y, txtValid_name_loss)
    txt_loss_valid_psnr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_Valid_psnr_Y.txt'
    txt_valid_psnr_path_Y = os.path.join(logs_path_Y, txt_loss_valid_psnr)

    txt_loss_mse = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))+'_loss_U.txt'
    txt_mse_path_U = os.path.join(logs_path_U, txt_loss_mse)
    txt_loss_psnr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))+'_psnr_U.txt'
    txt_psnr_path_U = os.path.join(logs_path_U, txt_loss_psnr)
    txtValid_name_loss = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))+'_Valid_loss_U.txt'
    txt_lossValid_path_U = os.path.join(logs_path_U, txtValid_name_loss)
    txt_loss_valid_psnr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_Valid_psnr_U.txt'
    txt_valid_psnr_path_U = os.path.join(logs_path_U, txt_loss_valid_psnr)

    txt_loss_mse = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_loss_V.txt'
    txt_mse_path_V = os.path.join(logs_path_V, txt_loss_mse)
    txt_loss_psnr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_psnr_V.txt'
    txt_psnr_path_V = os.path.join(logs_path_V, txt_loss_psnr)
    txtValid_name_loss = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_Valid_loss_V.txt'
    txt_lossValid_path_V = os.path.join(logs_path_V, txtValid_name_loss)
    txt_loss_valid_psnr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_Valid_psnr_V.txt'
    txt_valid_psnr_path_V = os.path.join(logs_path_V, txt_loss_valid_psnr)

    criterion = nn.MSELoss()
    #
    # optimizer = optim.Adam([
    #     {'params': model.base.parameters()},
    #     {'params': model.last.parameters(), 'lr': opt.lr * 0.1},
    # ], lr=opt.lr)
    optimizer_Y = optim.Adam(params=model1.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    optimizer_U = optim.Adam(params=model2.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    optimizer_V = optim.Adam(params=model3.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    # lr_init = optimizer.param_groups[0]['lr']
    if opt.resume == True:
        checkpoint = torch.load(opt.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model1.load_state_dict(checkpoint['model_state_dict'])
        optimizer_Y.load_state_dict(checkpoint['optimizer_state_dict'])
        model2.load_state_dict(checkpoint['model_state_dict'])
        optimizer_U.load_state_dict(checkpoint['optimizer_state_dict'])
        model3.load_state_dict(checkpoint['model_state_dict'])
        optimizer_V.load_state_dict(checkpoint['optimizer_state_dict'])

    train_dataset = h5Dataset(txtPth=opt.data_path)
    train_dataloader = DataLoader(dataset=train_dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=False,
                            drop_last=True)
    valid_dataset = h5Dataset(txtPth=opt.validData_path)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.threads,
                                  pin_memory=False,
                                  drop_last=True)

    model1.train()
    model2.train()
    model3.train()
    max_y = -1
    max_u = -1
    max_v = -1
    for epoch in range(opt.num_epochs):
        epoch_losses_Y = AverageMeter()
        epoch_losses_ori_Y = AverageMeter()
        epoch_losses_U = AverageMeter()
        epoch_losses_ori_U = AverageMeter()
        epoch_losses_V = AverageMeter()
        epoch_losses_ori_V = AverageMeter()
        adjust_learning_rate(optimizer_Y, epoch, opt.lr)
        adjust_learning_rate(optimizer_U, epoch, opt.lr)
        adjust_learning_rate(optimizer_V, epoch, opt.lr)

        with tqdm(total=(len(train_dataset) - len(train_dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch, opt.num_epochs))
            for data in train_dataloader:
                inputs, labels = data         # [B,1,32,32]
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
                is_flip = random.randrange(0, 2)
                if is_flip:
                    inputs = torch.flip(inputs, [2, 3])
                    labels = torch.flip(labels, [2, 3])
                inputs = inputs.to(device)
                labels = labels.to(device)
                # inputs_2D=projection(inputs)
                # preds_Y = model1(torch.unsqueeze(inputs[:, 0, :, :], -3), 0)   # [B,3,32,32]
                # preds_U = model2(torch.unsqueeze(inputs[:, 1, :, :], -3), 1)  # [B,3,32,32]
                # preds_V = model3(torch.unsqueeze(inputs[:, 2, :, :], -3), 2)  # [B,3,32,32]
                preds_Y = model1(torch.unsqueeze(inputs[:, 0, :, :], -3))  # [B,3,32,32]
                preds_U = model2(torch.unsqueeze(inputs[:, 1, :, :], -3))  # [B,3,32,32]
                preds_V = model3(torch.unsqueeze(inputs[:, 2, :, :], -3))  # [B,3,32,32]

                #preds_3D=inv_projection(preds)
                loss_Y = criterion(preds_Y, torch.unsqueeze(labels[:, 0, :, :], -3))
                loss_origin_Y = criterion(torch.unsqueeze(inputs[:, 0, :, :], -3),
                                          torch.unsqueeze(labels[:, 0, :, :], -3))
                loss_U = criterion(preds_U, torch.unsqueeze(labels[:, 1, :, :], -3))
                loss_origin_U = criterion(torch.unsqueeze(inputs[:, 1, :, :], -3),
                                          torch.unsqueeze(labels[:, 1, :, :], -3))
                loss_V = criterion(preds_V, torch.unsqueeze(labels[:, 2, :, :], -3))
                loss_origin_V = criterion(torch.unsqueeze(inputs[:, 2, :, :], -3),
                                          torch.unsqueeze(labels[:, 2, :, :], -3))
                # loss = criterion(preds, labels)
                # loss_origin = criterion(inputs, labels)
                li = len(inputs)
                epoch_losses_Y.update(loss_Y.item(), li)
                epoch_losses_ori_Y.update(loss_origin_Y.item(), li)
                epoch_losses_U.update(loss_U.item(), li)
                epoch_losses_ori_U.update(loss_origin_U.item(), li)
                epoch_losses_V.update(loss_V.item(), li)
                epoch_losses_ori_V.update(loss_origin_V.item(), li)

                optimizer_Y.zero_grad()
                loss_Y.backward()
                optimizer_Y.step()
                optimizer_U.zero_grad()
                loss_U.backward()
                optimizer_U.step()
                optimizer_V.zero_grad()
                loss_V.backward()
                optimizer_V.step()

                _tqdm.set_postfix(loss='{:.7f}'.format(epoch_losses_Y.avg))
                _tqdm.update(li)
        # writer.add_scalar('train_loss', epoch_losses.avg, epoch)
        epoch_psnr = mse2psnr(epoch_losses_Y.avg)
        epoch_psnr_ori = mse2psnr(epoch_losses_ori_Y.avg)
        file_log = open(txt_mse_path_Y, 'a')
        print('epoch:{} loss:{} loss_ori:{}'.format(epoch, epoch_losses_Y.avg, epoch_losses_ori_Y.avg),
              file=file_log)
        file_log.close()
        print(datetime.datetime.now())
        print('epoch:{}'.format(epoch), 'average loss:{}'.format(epoch_losses_Y.avg))
        file_loss = open(txt_psnr_path_Y, 'a')
        print('epoch:{}'.format(epoch), 'psnr:{}'.format(epoch_psnr), 'psnr_origin:{}'.format(epoch_psnr_ori),
              file=file_loss)
        file_loss.close()

        epoch_psnr = mse2psnr(epoch_losses_U.avg)
        epoch_psnr_ori = mse2psnr(epoch_losses_ori_U.avg)
        file_log = open(txt_mse_path_U, 'a')
        print('epoch:{} loss:{} loss_ori:{}'.format(epoch, epoch_losses_U.avg, epoch_losses_ori_U.avg),
              file=file_log)
        file_log.close()
        print(datetime.datetime.now())
        print('epoch:{}'.format(epoch), 'average loss:{}'.format(epoch_losses_U.avg))
        file_loss = open(txt_psnr_path_U, 'a')
        print('epoch:{}'.format(epoch), 'psnr:{}'.format(epoch_psnr), 'psnr_origin:{}'.format(epoch_psnr_ori),
              file=file_loss)
        file_loss.close()

        epoch_psnr = mse2psnr(epoch_losses_V.avg)
        epoch_psnr_ori = mse2psnr(epoch_losses_ori_V.avg)
        file_log = open(txt_mse_path_V, 'a')
        print('epoch:{} loss:{} loss_ori:{}'.format(epoch, epoch_losses_V.avg, epoch_losses_ori_V.avg),
              file=file_log)
        file_log.close()
        print(datetime.datetime.now())
        print('epoch:{}'.format(epoch), 'average loss:{}'.format(epoch_losses_V.avg))
        file_loss = open(txt_psnr_path_V, 'a')
        print('epoch:{}'.format(epoch), 'psnr:{}'.format(epoch_psnr), 'psnr_origin:{}'.format(epoch_psnr_ori),
              file=file_loss)
        file_loss.close()



         # start validating...

        print('epoch:%d   validating...' % epoch)
        model1.eval()
        model2.eval()
        model3.eval()
        valid_epoch_loss_Y = AverageMeter()
        valid_epoch_loss_ori_Y = AverageMeter()
        valid_epoch_loss_U = AverageMeter()
        valid_epoch_loss_ori_U = AverageMeter()
        valid_epoch_loss_V = AverageMeter()
        valid_epoch_loss_ori_V = AverageMeter()
        with tqdm(total=(len(valid_dataset) - len(valid_dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch, opt.num_epochs))
            for data in valid_dataloader:
                inputs, labels = data         # [B,3,1024]
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    # preds_Y = model1(torch.unsqueeze(inputs[:, 0, :, :], -3), 0)  # [B,3,1024]
                    # preds_U = model2(torch.unsqueeze(inputs[:, 1, :, :], -3), 1)  # [B,3,1024]
                    # preds_V = model3(torch.unsqueeze(inputs[:, 2, :, :], -3), 2)  # [B,3,1024]
                    preds_Y = model1(torch.unsqueeze(inputs[:, 0, :, :], -3))  # [B,3,1024]
                    preds_U = model2(torch.unsqueeze(inputs[:, 1, :, :], -3))  # [B,3,1024]
                    preds_V = model3(torch.unsqueeze(inputs[:, 2, :, :], -3))  # [B,3,1024]
                loss_Y = criterion(preds_Y, torch.unsqueeze(labels[:, 0, :, :], -3))
                loss_origin_Y = criterion(torch.unsqueeze(inputs[:, 0, :, :], -3),
                                          torch.unsqueeze(labels[:, 0, :, :], -3))
                loss_U = criterion(preds_U, torch.unsqueeze(labels[:, 1, :, :], -3))
                loss_origin_U = criterion(torch.unsqueeze(inputs[:, 1, :, :], -3),
                                          torch.unsqueeze(labels[:, 1, :, :], -3))
                loss_V = criterion(preds_V, torch.unsqueeze(labels[:, 2, :, :], -3))
                loss_origin_V = criterion(torch.unsqueeze(inputs[:, 2, :, :], -3),
                                          torch.unsqueeze(labels[:, 2, :, :], -3))
                # loss = criterion(preds, labels)
                valid_epoch_loss_Y.update(loss_Y.item(), len(inputs))
                valid_epoch_loss_ori_Y.update(loss_origin_Y.item(), len(inputs))
                valid_epoch_loss_U.update(loss_U.item(), len(inputs))
                valid_epoch_loss_ori_U.update(loss_origin_U.item(), len(inputs))
                valid_epoch_loss_V.update(loss_V.item(), len(inputs))
                valid_epoch_loss_ori_V.update(loss_origin_V.item(), len(inputs))

                _tqdm.set_postfix(loss='{:.7f}'.format(valid_epoch_loss_Y.avg))
                _tqdm.update(len(inputs))
        # writer.add_scalar('test_loss', valid_epoch_loss.avg, epoch)
        epoch_valid_psnr = mse2psnr(valid_epoch_loss_Y.avg)
        epoch_valid_psnr_ori = mse2psnr(valid_epoch_loss_ori_Y.avg)
        print('valid loss_ori:{}'.format(valid_epoch_loss_ori_Y.avg),
              'valid loss_preds:{}'.format(valid_epoch_loss_Y.avg))
        fileValid_loss = open(txt_lossValid_path_Y, 'a')
        print('epoch:{}'.format(epoch), 'valid average loss:{}'.format(valid_epoch_loss_Y.avg),
              'valid average loss_ori:{}'.format(valid_epoch_loss_ori_Y.avg), file=fileValid_loss)
        fileValid_loss.close()
        file_loss = open(txt_valid_psnr_path_Y, 'a')
        print('epoch:{}'.format(epoch), 'psnr:{}'.format(epoch_valid_psnr), 'psnr_origin:{}'.format(epoch_valid_psnr_ori),
              file=file_loss)
        file_loss.close()
        if epoch_valid_psnr > max_y:
            max_y = epoch_valid_psnr
            torch.save({
                "epoch": epoch,
                "model_state_dict": model1.state_dict(),
                "optimizer_state_dict": optimizer_Y.state_dict(),
                "loss": loss_Y,
            },
                '%s/model_%d.pth' % (model_path_Y, epoch)
            )

        epoch_valid_psnr = mse2psnr(valid_epoch_loss_U.avg)
        epoch_valid_psnr_ori = mse2psnr(valid_epoch_loss_ori_U.avg)
        print('valid loss_ori:{}'.format(valid_epoch_loss_ori_U.avg),
              'valid loss_preds:{}'.format(valid_epoch_loss_U.avg))
        fileValid_loss = open(txt_lossValid_path_U, 'a')
        print('epoch:{}'.format(epoch), 'valid average loss:{}'.format(valid_epoch_loss_U.avg),
              'valid average loss_ori:{}'.format(valid_epoch_loss_ori_U.avg), file=fileValid_loss)
        fileValid_loss.close()
        file_loss = open(txt_valid_psnr_path_U, 'a')
        print('epoch:{}'.format(epoch), 'psnr:{}'.format(epoch_valid_psnr),
              'psnr_origin:{}'.format(epoch_valid_psnr_ori),
              file=file_loss)
        file_loss.close()

        if epoch_valid_psnr > max_u:
            max_u = epoch_valid_psnr
            torch.save({
                "epoch": epoch,
                "model_state_dict": model2.state_dict(),
                "optimizer_state_dict": optimizer_U.state_dict(),
                "loss": loss_U,
            },
                '%s/model_%d.pth' % (model_path_U, epoch)
            )

        epoch_valid_psnr = mse2psnr(valid_epoch_loss_V.avg)
        epoch_valid_psnr_ori = mse2psnr(valid_epoch_loss_ori_V.avg)
        print('valid loss_ori:{}'.format(valid_epoch_loss_ori_V.avg),
              'valid loss_preds:{}'.format(valid_epoch_loss_V.avg))
        fileValid_loss = open(txt_lossValid_path_V, 'a')
        print('epoch:{}'.format(epoch), 'valid average loss:{}'.format(valid_epoch_loss_V.avg),
              'valid average loss_ori:{}'.format(valid_epoch_loss_ori_V.avg), file=fileValid_loss)
        fileValid_loss.close()
        file_loss = open(txt_valid_psnr_path_V, 'a')
        print('epoch:{}'.format(epoch), 'psnr:{}'.format(epoch_valid_psnr),
              'psnr_origin:{}'.format(epoch_valid_psnr_ori),
              file=file_loss)
        file_loss.close()
        if epoch_valid_psnr > max_v:
            max_v = epoch_valid_psnr
            torch.save({
                "epoch": epoch,
                "model_state_dict": model3.state_dict(),
                "optimizer_state_dict": optimizer_V.state_dict(),
                "loss": loss_V,
            },
                '%s/model_%d.pth' % (model_path_V, epoch)
            )




    # writer.close()
        # torch.save(model.state_dict(), os.path.join(opt.outputs_dir, '{}_epoch_{}.pth'.format(opt.arch, epoch)))
