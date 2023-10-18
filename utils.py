import numpy as np
import math
import torch
from numpy import *
from plyfile import PlyData, PlyElement


class AverageMeter(object):
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


def mse2psnr(mse):
    psnr=10*math.log10(255.0*255/mse)
    return psnr


def rgb2yuv(rgb):
    # PointNum=rgb.shape[0]
    yuv=np.zeros(rgb.shape)
    yuv[:, 0] = 0.2126*rgb[:, 0]+0.7152*rgb[:, 1]+0.0722*rgb[:, 2]
    yuv[:, 1]=-0.1146*rgb[:, 0]-0.3854*rgb[:, 1]+0.5000*rgb[:, 2] + 128
    yuv[:, 2]=0.5000*rgb[:, 0]-0.4542*rgb[:, 1]-0.0458*rgb[:, 2] + 128
    # for i in range(PointNum):
    #     yuv[i, 0]=0.2126*rgb[i,0]+0.7152*rgb[i,1]+0.0722*rgb[i,2];
    #     yuv[i, 1]=-0.1146*rgb[i,0]-0.3854*rgb[i,1]+0.5000*rgb[i,2]+128;
    #     yuv[i, 2]=0.5000*rgb[i,0]-0.4542*rgb[i,1]-0.0458*rgb[i,2]+128;
    yuv=yuv.astype(np.float32)
    return yuv


def yuv2rgb(yuv):
    # PointNum=yuv.shape[0]
    yuv[:, 1] = yuv[:, 1] - 128
    yuv[:, 2] = yuv[:, 2] - 128
    rgb=np.zeros(yuv.shape)
    rgb[:, 0] = yuv[:, 0] + 1.57480 * yuv[:, 2]
    rgb[:, 1] = yuv[:, 0] - 0.18733 * yuv[:, 1] - 0.46813 * yuv[:, 2]
    rgb[:, 2] = yuv[:, 0] + 1.85563 * yuv[:, 1]
    # for i in range(PointNum):
    #     rgb[i, 0]=0.2126*yuv[i,0]+0.7152*yuv[i,1]+0.0722*yuv[i,2];
    #     rgb[i, 1]=-0.1146*yuv[i,0]-0.3854*yuv[i,1]+0.5000*yuv[i,2]+128;
    #     rgb[i, 2]=0.5000*yuv[i,0]-0.4542*yuv[i,1]-0.0458*yuv[i,2]+128;
    rgb=rgb.astype(np.float32)
    return rgb



def cal_psnr(input1, input2):
    input1 = input1
    input2 = input2
    # img1 = input1.astype(np.float64)
    # img2 = input2.astype(np.float64)
    img1 = input1.to(torch.float64)
    img2 = input2.to(torch.float64)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(255.0*255 / math.sqrt(mse))
    return psnr


def read_ply(filename):
    """ read XYZ (RGB)point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z, red, green, blue] for x, y, z, red, green, blue in pc[['x', 'y', 'z', 'red', 'green', 'blue']]])
    return pc_array


def write_ply(points, filename, text=True):
    """ input: Nx3(6), write points to filename as PLY format. """
    points = [(points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4], points[i, 5]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
   #  vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    xyz = torch.tensor(xyz)    # from numpy to tensor
    xyz = xyz.to(torch.float)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    centroids = centroids.detach().numpy()      # from tensor to numpy
    return centroids


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def knn(centroid_xyz, xyz, npoint):
        """ 计算欧式距离,计算centroid_xyz中的每个点在xyz中的npoint个邻近点"""
        centroid_xyz = centroid_xyz.astype(np.float32)
        xyz = xyz.astype(np.float32)
        size_centroid = centroid_xyz.shape[0]  # 训练样本量大小, numPatch
        size_xyz = xyz.shape[0]  # 测试样本大小 , pointNum
        XX = centroid_xyz ** 2
        sumXX = np.sum(XX, axis=1, keepdims=True)  # 行平方和
        YY = xyz ** 2
        sumYY = np.sum(YY, axis=1, keepdims=False)  # 行平方和
        # Xpw2_plus_Ypw2 = sumXX.repeat(1, size_xyz) + sumYY.repeat(size_centroid, 1)
        Xpw2_plus_Ypw2 = tile(sumXX, [1, size_xyz]) + tile(sumYY, [size_centroid, 1])
        EDsq = Xpw2_plus_Ypw2 - 2 * np.dot(centroid_xyz, xyz.T)  # 欧式距离平方
        # EDsq = Xpw2_plus_Ypw2 - 2 * torch.mm(centroid_xyz, xyz.permute(1, 0))
        # distances = EDsq.sqrt()
        distances = array(EDsq) ** 0.5  # 欧式距离 [size_centroid, size_xyz]
        # _, group_idxs = torch.sort(distances)
        group_idxs = np.argsort(distances)    # [size_centroid, size_xyz]
        group_idx = group_idxs[:, 0:npoint]    # [size_centroid, npoint]
        return group_idx


def k_nearest(centroid_xyz, xyz, npoint):
    centroid_xyz = centroid_xyz.astype(np.float32)
    xyz = xyz.astype(np.float32)
    size_centroid = centroid_xyz.shape[0]
    size_xyz = xyz.shape[0]
    idx = []
    for i in range(size_centroid):
        curLoc = centroid_xyz[i, :]
        diffMat = tile(curLoc, (size_xyz, 1)) - xyz
        sqdiffMat = diffMat ** 2                            # [size_xyz, 3]
        sqDistances = sqdiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        sortedDistIndicies = distances.argsort()
        idx_npoint = sortedDistIndicies[0:npoint]
        idx.append(idx_npoint)
        group_idx = np.array(idx)
    return group_idx


def matchLoc(rec_loc, ori):
    """ 由于经过编码器编码后点的顺序将打乱，该函数就是根据重建点云中点的顺序重新排列原始点云的顺序，输出为重排序后的原始点云序列
    rec_loc: numPoint x 3, ori: numPoint x 6, output: numPoint x 6, """
    numPoint = rec_loc.shape[0]   # point number
    pt_new = np.ones((numPoint, 6))
    ori_loc = ori[:, 0:3]
    ori_locList = ori_loc.tolist()
    for i in range(numPoint):
        curLoc = rec_loc[i, :]
        idx = ori_locList.index(curLoc.tolist())
        pt_new[i, :] = ori[idx, :]
    return pt_new


def cal_mean(list):       # 对于重复使用的点计算加权平均值
    number = len(list)
    idx = [index for index in range(number) if list[index].size != 1]       #  找出重复使用的点的索引
    for i in idx:
        list[i] = np.mean(list[i], axis=0)
    return list


def rectangular(input):    # 将输入按照回字形排列, input:[1024,3] --> output:[3, 32, 32]
    pointNum = input.shape[0]    # 输入是按照距离第一个点由近及远的顺序排列的
    n = int(np.sqrt(pointNum))
    upb = 0  # upper bound
    lob = n-1  # lower bound
    lfb = 0  # left  bound
    rtb = n-1  # right bound
    j = 0
    i = -1
    data = np.zeros((1, n, n), dtype=np.float32)
    r = np.zeros((n, n), dtype=np.float32)
    g = np.zeros((n, n), dtype=np.float32)
    b = np.zeros((n, n), dtype=np.float32)
    while pointNum > 0:
        while i < lob:
            i = i + 1
            r[i, j] = input[pointNum-1]
            # g[i, j] = input[pointNum-1, 1]
            # b[i, j] = input[pointNum-1, 2]
            pointNum = pointNum - 1
        lob = lob - 1
        while j < rtb:
            j = j + 1
            r[i, j] = input[pointNum-1]
            # g[i, j] = input[pointNum-1, 1]
            # b[i, j] = input[pointNum-1, 2]
            pointNum = pointNum - 1
        rtb = rtb - 1
        while i > upb:
            i = i - 1
            r[i, j] = input[pointNum-1]
            # g[i, j] = input[pointNum-1, 1]
            # b[i, j] = input[pointNum-1, 2]
            pointNum = pointNum - 1
        upb = upb + 1
        while j > lfb + 1:
            j = j - 1
            r[i, j] = input[pointNum-1]
            # g[i, j] = input[pointNum-1, 1]
            # b[i, j] = input[pointNum-1, 2]
            pointNum = pointNum - 1
        lfb = lfb + 1
    data[0, :, :] = r
    # data[1, :, :] = g
    # data[2, :, :] = b
    return data


def de_rectangular(input):     # input:[3,32,32] --> output:[1024,3]
    length = input.shape[1]    # 32
    npoint = np.square(length)    # point number
    centroids = np.zeros((npoint, 1), dtype=np.float32)
    upb = 0  # upper bound
    lob = length - 1  # lower bound
    lfb = 0  # left  bound
    rtb = length - 1  # right bound
    j = 0
    i = -1
    while npoint > 0:
        while i < lob:
            i = i + 1
            centroids[npoint-1, 0] = input[i, j]
            # centroids[npoint-1, 1] = input[1, i, j]
            # centroids[npoint-1, 2] = input[2, i, j]
            npoint = npoint - 1
        lob = lob - 1
        while j < rtb:
            j = j + 1
            centroids[npoint - 1, 0] = input[i, j]
            # centroids[npoint - 1, 1] = input[1, i, j]
            # centroids[npoint - 1, 2] = input[2, i, j]
            npoint = npoint - 1
        rtb = rtb - 1
        while i > upb:
            i = i - 1
            centroids[npoint - 1, 0] = input[i, j]
            # centroids[npoint - 1, 1] = input[1, i, j]
            # centroids[npoint - 1, 2] = input[2, i, j]
            npoint = npoint - 1
        upb = upb + 1
        while j > lfb + 1:
            j = j - 1
            centroids[npoint - 1, 0] = input[i, j]
            # centroids[npoint - 1, 1] = input[1, i, j]
            # centroids[npoint - 1, 2] = input[2, i, j]
            npoint = npoint - 1
        lfb = lfb + 1

    return centroids

def search_knn(c, x, k):
    pairwise_distance = torch.sum(torch.pow((c - x), 2), dim = -1)
    idx = (-pairwise_distance).topk(k = k, dim = -1)[1]   # (batch_size, num_points, k)
    return idx

if __name__ == '__main__':
    # color = np.array([[3.9922432, 4.23, 5], [2, 75, 23]])
    # color = color.astype(np.float32)
    # loc = np.array([[827, 98, 9], [92, 87, 3]])
    # filename = 'test.ply'
    # pt = np.concatenate((loc, color), axis=1)
    # write_ply(pt, filename)
    input_data = np.random.randn(4,4).astype(np.float32)
    out = de_rectangular(input_data)
