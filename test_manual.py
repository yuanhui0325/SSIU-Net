import argparse
import os
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
# from kd_tree import *
# from model_unet import ARCNN, ARCNN_Auto, VRCNN, VDSR, SE_VRCNN, U_Net
from model_new import ARCNN, ARCNN_Auto, VRCNN, VDSR, SE_VRCNN, SSIU_Net, PtRestoration, U_Net
from dataset import h5Dataset, h5DataSet_more
from utils import *
import warnings
import time
import datetime
from sewar.full_ref import psnr
warnings.filterwarnings("ignore")
#os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
c = '1'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='SSIU', help='which network')
    parser.add_argument('--log_path', type=str, default='./logs_test/2023/SSIU_LT_new/')
    parser.add_argument('--batchSize', type=int, default=150)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)
    # parser.add_argument('--data_path', type=str, default='./dataset_creat/data_ori_test_LT/testFile' + c +'.txt')
    parser.add_argument('--data_path', type=str, default='./dataset_creat/data_ori_test_LT/testFile.txt')
    parser.add_argument('--pretrain_Y', type=str, default='./pths/2023/SSIU_LT_new/Y/2023-10-11/R0'+c+'_SSIU_Net/model_20.pth')
    parser.add_argument('--pretrain_U', type=str, default='./pths/2023/SSIU_LT_new/U/2023-10-11/R0'+c+'_SSIU_Net/model_20.pth')
    parser.add_argument('--pretrain_V', type=str, default='./pths/2023/SSIU_LT_new/V/2023-10-11/R0'+c+'_SSIU_Net/model_10.pth')
    parser.add_argument('--ori_path', type=str, default='./dataset_creat/data_ori_test_LT/same_order/')
    parser.add_argument('--rec_path', type=str, default='./dataset_creat/data_rec_test_LT/')
    parser.add_argument('--pred_path', type=str, default='./data/preds/2023/SSIU_LT_new/LT_r0'+c+'_SSIU_LT_new/')
    parser.add_argument('--notes', type=str, default='LT_r0'+c+'_YUV_U_Net')
    return parser.parse_args()


def eval_new(opt, model, input, types=1):
    model.eval()
    preds = model(input, types)
    # preds = model(input)
    return preds


def log_string(log, out_str):
    log.write(out_str + '\n')
    log.flush()
    print(out_str)


def main(opt):
    if not os.path.exists(opt.log_path):
        os.makedirs(opt.log_path)
    textfile_name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_test_SSIU.txt'
    LOG_FOUT = open(os.path.join(opt.log_path, textfile_name), 'w')
    LOG_FOUT.write(str(opt) + '\n')
    model1 = SSIU_Net()
    model2 = SSIU_Net()
    model3 = SSIU_Net()
    checkpoint1 = torch.load(opt.pretrain_Y, map_location={'cuda:1': 'cuda:0'})
    checkpoint2 = torch.load(opt.pretrain_U, map_location={'cuda:1': 'cuda:0'})
    checkpoint3 = torch.load(opt.pretrain_V, map_location={'cuda:1': 'cuda:0'})
    model1.load_state_dict(checkpoint1['model_state_dict'])
    model2.load_state_dict(checkpoint2['model_state_dict'])
    model3.load_state_dict(checkpoint3['model_state_dict'])
    print('load models...')
    model1.to(device)
    model2.to(device)
    model3.to(device)
    iter = 0
    if not os.path.exists(opt.pred_path):
        os.makedirs(opt.pred_path)
    with open(opt.data_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            log_string(LOG_FOUT, line)
            iter = iter + 1
            log_string(LOG_FOUT, 'sequence: %s, iter: %d' % (line, iter))
            if os.path.splitext(line)[1] == ".ply":
                pointcloud_ori = read_ply(os.path.join(opt.ori_path, line))  # [numpoint, 6]
                ori_color = pointcloud_ori[:, 3:6]
                ori_color = rgb2yuv(ori_color)
                ori_color_y = ori_color[:, 0]
                ori_color_u = ori_color[:, 1]
                ori_color_v = ori_color[:, 2]
                pointcloud_rec = read_ply(os.path.join(opt.rec_path, os.path.splitext(line)[0]+'_r0'+c+'.ply'))
                # pointcloud_rec = read_ply(os.path.join(opt.rec_path, os.path.splitext(line)[0]+'_r01.ply'))
                rec_loc = pointcloud_rec[:, 0:3]    # [numpoint, 3],location
                rec_color = pointcloud_rec[:, 3:6]   # color
                rec_color = rgb2yuv(rec_color)
                # rec_color = rgb2yuv(rec_color_rgb)          # [pointNum,3]
                rec_color_y = rec_color[:, 0].astype(np.float32)            # extract the y component
                rec_color_u = rec_color[:, 1].astype(np.float32)
                rec_color_v = rec_color[:, 2].astype(np.float32)
                pointNum = rec_loc.shape[0]
                numPatch = pointNum // 512 + 1
                patch_seg_time1 = time.time()
                pt_locs = np.expand_dims(rec_loc, 0)        # [1,numpoint,3]
                idx = farthest_point_sample(pt_locs, numPatch)    # [1,numPatch]
                centroid_xyz = rec_loc[idx, :]                    # [1, numPatch, 3]
                centroid_xyz = np.squeeze(centroid_xyz)           # [numPatch, 3]
                # group_idx = knn(centroid_xyz, rec_loc, 1024)   # [numPatch, 1024]
                print("k nearest neighbor doing...")
                group_idx = torch.zeros((centroid_xyz.shape[0], 1024))
                for i in range(centroid_xyz.shape[0]):
                    group_idx[i, :] = search_knn(torch.tensor(centroid_xyz[i]), torch.tensor(rec_loc), 1024)
                # group_idx = search_knn(centroid_xyz, rec_loc, 1024)   # group_idx 中的每一个数的数据类型必须为整型
                # assert idx[0] == group_idx[:, 0]   # 第一个最邻近肯定是自己
                patch_seg_time2 = time.time()
                group_idx = np.array(group_idx, dtype=int)
                unique_idx = np.unique(group_idx)
                log_string(LOG_FOUT, str((len(unique_idx) / pointNum)))      # 打印出点云中点的利用率
                output_new = np.zeros((pointNum, 6))
                output_new[:, :3] = rec_loc
                new_color = np.zeros((pointNum, 3), dtype=np.float32)
                # new_color_y = new_color[:, 0]  # initialize
                new_color_y = list(new_color[:, 0])
                new_color_u = list(new_color[:, 1])
                new_color_v = list(new_color[:, 2])
                # new_color[:, 1:3] = rec_color[:, 1:3]     # uv分量直接赋值

                log_string(LOG_FOUT, "extract patches...%s" % datetime.datetime.now())
                # print("extract patches...{}".format(datetime.datetime.now()))
                process_time1 = time.time()
                for j in range(numPatch):
                    curIdx = group_idx[j, :]   # [1024,]
                    curColor_y = rec_color_y[curIdx].astype(np.float32)    # [1024, 1]
                    curColor_u = rec_color_u[curIdx].astype(np.float32)  # [1024, 1]
                    curColor_v = rec_color_v[curIdx].astype(np.float32)  # [1024, 1]
                    # 保证每一页上都是同一个分量R，G，或者B
                    # 保证数据是按照离代表点由近及远按列排序的,与matlab产生训练数据时的规则一样
                    # input = data.transpose(0, 2, 1) / 255.0
                    # data = np.expand_dims(curColor, axis=0)
                    data_y = rectangular(curColor_y)  # [1,32,32]
                    data_u = rectangular(curColor_u)  # [1,32,32]
                    data_v = rectangular(curColor_v)  # [1,32,32]
                    input_y = data_y
                    input_y = torch.tensor(input_y).unsqueeze(0).to(device)          # nd_array to tensor
                    outputs_y = eval_new(opt, model1, input_y, 0)                       # [1,3,32,32]
                    # device: cuda, tensor to ndarray
                    output_y = np.squeeze(outputs_y.cpu().detach().numpy())                                        # [3,32,32]
                    output_vec_y = de_rectangular(output_y)                      # [1024, 1]

                    input_u = data_u
                    input_u = torch.tensor(input_u).unsqueeze(0).to(device)  # nd_array to tensor
                    outputs_u = eval_new(opt, model2, input_u)  # [1,3,32,32]
                    output_u = np.squeeze(outputs_u.cpu().detach().numpy())  # [3,32,32]
                    output_vec_u = de_rectangular(output_u)  # [1024, 1]

                    input_v = data_v
                    input_v = torch.tensor(input_v).unsqueeze(0).to(device)  # nd_array to tensor
                    outputs_v = eval_new(opt, model3, input_v, 2)  # [1,3,32,32]
                    output_v = np.squeeze(outputs_v.cpu().detach().numpy())  # [3,32,32]
                    output_vec_v = de_rectangular(output_v)  # [1024, 1]

                    for t, m in enumerate(curIdx):
                        if (new_color_y[m] == 0).all():
                            new_color_y[m] = output_vec_y[t]
                            new_color_u[m] = output_vec_u[t]
                            new_color_v[m] = output_vec_v[t]
                        # 有重复的点，先concate，最后再计算均值
                        else:
                            new_color_y[m] = np.concatenate((new_color_y[m].reshape(-1, 1),
                                                             output_vec_y[t].reshape(-1, 1)), axis=0)
                            new_color_u[m] = np.concatenate((new_color_u[m].reshape(-1, 1),
                                                             output_vec_u[t].reshape(-1, 1)), axis=0)
                            new_color_v[m] = np.concatenate((new_color_v[m].reshape(-1, 1),
                                                             output_vec_v[t].reshape(-1, 1)), axis=0)
                # 对于重复选的点，计算均值作为最终的预测值
                process_time2 = time.time()
                output_color_y = np.array(cal_mean(new_color_y))     # from list to array()
                output_color_u = np.array(cal_mean(new_color_u))
                output_color_v = np.array(cal_mean(new_color_v))
                mask = output_color_y == 0          # 对于没有选中的点，直接赋值为重建点的值
                output_color_y = output_color_y.reshape(-1, 1)
                output_color_u = output_color_u.reshape(-1, 1)
                output_color_v = output_color_v.reshape(-1, 1)
                rec_color_y = rec_color_y.reshape(-1, 1)
                rec_color_u = rec_color_u.reshape(-1, 1)
                rec_color_v = rec_color_v.reshape(-1, 1)
                output_color_y[mask] = rec_color_y[mask]     # [pointNum, 1]
                output_color_u[mask] = rec_color_u[mask]
                output_color_v[mask] = rec_color_v[mask]
                output_color_y = np.squeeze(output_color_y)        # (pointNum)
                output_color_u = np.squeeze(output_color_u)
                output_color_v = np.squeeze(output_color_v)
                new_color[:, 0] = output_color_y
                new_color[:, 1] = output_color_u
                new_color[:, 2] = output_color_v
                new_color = yuv2rgb(new_color)
                output_new[:, 3:6] = np.clip(np.round(new_color), 0, 255)
                filepath = os.path.join(opt.pred_path, os.path.splitext(line)[0]+'_r0'+c+'.ply')
                # filepath = os.path.join(opt.pred_path, os.path.splitext(line)[0]+'_r01.ply')
                write_ply(output_new, filepath)
                rec_color_y = np.squeeze(rec_color_y)
                rec_color_u = np.squeeze(rec_color_u)
                rec_color_v = np.squeeze(rec_color_v)
                patch_fuse_time = time.time()
                psnr_ori_y = psnr(ori_color_y, rec_color_y, MAX=255)
                psnr_pred_y = psnr(ori_color_y, output_color_y, MAX=255)
                psnr_ori_u = psnr(ori_color_u, rec_color_u, MAX=255)
                psnr_pred_u = psnr(ori_color_u, output_color_u, MAX=255)
                psnr_ori_v = psnr(ori_color_v, rec_color_v, MAX=255)
                psnr_pred_v = psnr(ori_color_v, output_color_v, MAX=255)
                log_string(LOG_FOUT, "psnr_y for rec: %f" % psnr_ori_y)
                log_string(LOG_FOUT, "psnr_y for pred: %f" % psnr_pred_y)

                log_string(LOG_FOUT, "psnr_u for rec: %f" % psnr_ori_u)
                log_string(LOG_FOUT, "psnr_u for pred: %f" % psnr_pred_u)

                log_string(LOG_FOUT, "psnr_v for rec: %f" % psnr_ori_v)
                log_string(LOG_FOUT, "psnr_v for pred: %f" % psnr_pred_v)

                log_string(LOG_FOUT, "\n patch_seg time:  %f" % (patch_seg_time2 - patch_seg_time1))
                log_string(LOG_FOUT, "processing time:  %f" % (process_time2 - process_time1))
                log_string(LOG_FOUT, "patch_fuse time:  %f \n \n" % (patch_fuse_time - process_time2))

    LOG_FOUT.close()


if __name__ == '__main__':
    opt = parse_args()
    main(opt)

