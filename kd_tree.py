# kd_Tree
# Edited By ocean_waver
import numpy as np
from utils import read_ply
import pandas as pd
#import matplotlib.pyplot as plt
"""
参数说明：
self.mid 						# 节点索引（中位数）
self.left						# 节点左空间索引列表
self.right = right				# 节点右空间索引列表
self.bound = bound  # Dim * 2	# 当前节点所在空间范围（每个维度由左右边界控制）
self.flag = flag				# 表示该节点对应的分割线应分割的维度索引（通过取模来控制变化）
self.lchild = lchild			# 左子节点地址
self.rchild = rchild			# 右子节点地址
self.par = par					# 父节点地址
self.l_bound = l_bound			# 节点左空间范围
self.r_bound = r_bound			# 节点右空间范围
self.side = side				# 当前节点是其父节点的左节点(0)或右节点(1)
"""

class Node(object):

    def __init__(self, mid, left, right, bound, flag, lchild=None, rchild=None, par=None,
                 l_bound=None, r_bound=None, side=-1):
        self.mid = mid
        self.left = left
        self.right = right
        self.bound = bound  # Dim * 2
        self.flag = flag
        self.lchild = lchild
        self.rchild = rchild
        self.par = par
        self.l_bound = l_bound
        self.r_bound = r_bound
        self.side = side


def find_median(a):
    # s = np.sort(a)
    arg_s = np.argsort(a)
    idx_mid = arg_s[len(arg_s) // 2]
    idx_left = np.array([arg_s[j] for j in range(0, len(arg_s) // 2)], dtype='int32')
    idx_right = np.array([arg_s[j] for j in range(len(arg_s) // 2 + 1, np.size(a))], dtype='int32')

    return idx_mid, idx_left, idx_right


def kd_tree_establish(root, points, dim):
    # print(root.mid)
    layer_flag = (root.flag + 1) % dim  # 确定分割点对应的分割线的维度

    if dim == 2:
        static_pos = points[root.mid, root.flag]
        if root.flag == 0:
            x_line = np.linspace(static_pos, static_pos, 10)
            y_line = np.linspace(root.bound[1, 0], root.bound[1, 1], 10)
        elif root.flag == 1:
            x_line = np.linspace(root.bound[0, 0], root.bound[0, 1], 10)
            y_line = np.linspace(static_pos, static_pos, 10)
       # plt.plot(x_line, y_line, color='darkorange')
        # plt.axis([0, 1, 0, 1])
        # plt.draw()
        # plt.pause(0.05)

    # new bound:
    root.l_bound = root.bound.copy()  # 先复制一份根节点边界(Note: need to use deep copy!)
    root.l_bound[root.flag, 1] = points[root.mid, root.flag]  # 改变特定边界的最大值，获取新边界
    root.r_bound = root.bound.copy()
    root.r_bound[root.flag, 0] = points[root.mid, root.flag]  # 改变特定边界的最小值，获取新边界

    if root.left.size > 0:
        # print('left : ', root.left)
        mid, left, right = find_median(points[root.left, layer_flag])
        mid, left, right = root.left[mid], root.left[left], root.left[right]

        left_node = Node(mid, left, right, root.l_bound, layer_flag)
        root.lchild = left_node
        left_node.par = root
        left_node.side = 0
        kd_tree_establish(left_node, points, dim)

    if root.right.size > 0:
        # print('right : ', root.right)
        mid, left, right = find_median(points[root.right, layer_flag])
        mid, left, right = root.right[mid], root.right[left], root.right[right]

        right_node = Node(mid, left, right, root.r_bound, layer_flag)
        root.rchild = right_node
        right_node.par = root
        right_node.side = 1
        kd_tree_establish(right_node, points, dim)


def distance(a, b, p):
    """
    Lp distance:
    input: a and b must have equal length
           p must be a positive integer, which decides the type of norm
    output: Lp distance of vector a-b"""
    try:
        vector = a - b
    except ValueError:
        print('Distance : input error !\n the coordinates have different length !')
    dis = np.power(np.sum(np.power(vector, p)), 1 / p)
    return dis


# def search_other_branch(target, branch_node, points, dim):


def judge_cross(circle, branch, dim):
    """
    Judge if a sphere in dimension(dim) and the space of the other branch cross each other
    cross     : return 1
    not cross : return 0"""
    # print(circle, branch)
    count = 0
    for j in range(0, dim):
        if circle[j, 1] < branch[j, 0] or circle[j, 0] > branch[j, 1]:
            count = count + 1
    if count == 0:
        return 1  # cross
    else:
        return 0


def search_knn(targets, Points, K):
    Num = Points.shape[0]
    Dim = Points.shape[1]
    p = 2          #compute the euclidean distance
    target_num = targets.shape[0]
    Mid, Left, Right = find_median(Points[:, 0])
    L_bound = np.min(Points, axis=0)
    R_bound = np.max(Points, axis=0)
    Bound = np.vstack((L_bound, R_bound)).T

    Root = Node(Mid, Left, Right, Bound, flag=0)
    print('kdTree establish ...')
    kd_tree_establish(Root, Points, Dim)
    print('kdTree establish Done')
    idx = np.zeros((target_num, K),dtype=int)         # save the index of each target point's k nearest neighbor
    # dis = np.zeros((target_num, K), dtype=float)
    for t in range(0, target_num):
        Target = targets[t, :]
        # print('current target:', Target)
        # 定位初始搜索区域
        node = Root
        temp = Root
        side = 0  # 下降定位在终止时点所在的是左侧(side=0)还是右侧(side=1)
        while temp is not None:
            if Points[temp.mid, temp.flag] > Target[temp.flag]:  # 大于的情况
                node = temp
                temp = temp.lchild
                side = 0
            else:  # 包括小于和等于的情况
                node = temp
                temp = temp.rchild
                side = 1
       #  print('start node : ', node.mid, Points[node.mid])

        # 搜索最近邻点
        can_idx = np.array([], dtype='int32')
        can_dis = np.array([])

        temp = node
        while node is not None:
            # min_dis = distance(Target, Points[can_idx[-1]])
            search_flag = False
            temp_dis = distance(Target, Points[node.mid], 2)

            if can_idx.size < K:  # 候选点列表未满
                can_idx = np.append(can_idx, node.mid)
                can_dis = np.append(can_dis, temp_dis)
            elif temp_dis < np.max(can_dis):
                can_idx[np.argmax(can_dis)] = node.mid
                can_dis[np.argmax(can_dis)] = temp_dis

            search_flag = False  # 查看另一支路是否为空
            if side == 0 and node.rchild is not None:
                branch_bound = node.rchild.bound
                branch_list = node.right
                search_flag = True
            elif side == 1 and node.lchild is not None:
                branch_bound = node.lchild.bound
                branch_list = node.left
                search_flag = True

            if search_flag is True:  # 开始判断和搜索另一侧的支路
                r = np.max(can_dis)
                # 构建Dim维球体边界
                temp_bound = np.array([[Target[i] - r, Target[i] + r] for i in range(0, Dim)])

                if judge_cross(temp_bound, branch_bound, Dim) == 1:  # 高维球与支路空间存在交叉

                    for i in branch_list:
                        a_dis = distance(Target, Points[i], 2)
                        if can_idx.size < K:  # 候选未满，直接添加
                            can_idx = np.append(can_idx, i)
                            can_dis = np.append(can_dis, a_dis)
                        elif a_dis < np.max(can_dis):  # 候选已满，更近者替换候选最远者
                            can_idx[np.argmax(can_dis)] = i
                            can_dis[np.argmax(can_dis)] = a_dis
                # 向上更新查找节点
            temp = node
            side = temp.side  # 更新刚离开的node所处的左右方位
            node = node.par

        # 输出结果
        sort_idx = np.argsort(can_dis)
        can_idx = can_idx[sort_idx]
        can_dis = can_dis[sort_idx]
        # print('candidate_index :    ', can_idx)
        # print('candidate_distance : ', np.round(can_dis, 4))
        idx[t, :] = can_idx
        # dis[t, :] = can_dis
    return idx


if __name__ == '__main__':

    # --------基本参数设置--------
    #Points = np.random.rand(Num, Dim) + 100  # 产生随机点
    # Points = np.array([[127,163,255],[126,165,255],[127,164,255],[127,165,254],[127,165,255],[127,167,253],[126,166,255],[126,167,254]])
    # Points = np.array([[127, 123, 134],[125, 128, 129],[129, 244, 242],[134, 137, 125],[123, 245, 632],[133, 982, 134],[134, 156, 176], [173, 142, 131]])
    Points = read_ply('longdress_vox10_1300.ply')
    # Target = np.array([[127, 123, 134], [134, 137, 125]])     # 设定目标查询点
    Target = Points[[0,1], 0:3]
    k = 1024
    [idx, dis] = search_knn(Target, Points[:, 0:3], k)
    print('done')
    data = pd.DataFrame(idx)

    writer = pd.ExcelWriter('A.xlsx')  # 写入Excel文件
    data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    writer.save()
    writer.close()
    data1 = pd.DataFrame(dis)
    writer = pd.ExcelWriter('B.xlsx')  # 写入Excel文件
    data1.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    writer.save()
    writer.close()
     # Target = np.squeeze(np.random.rand(1, Dim))  # 这里只考虑一个目标点

    '''# Test for find_median()
    idx_mid, idx_left, idx_right = find_median(Points[:, 0])
    print(Points[:, 0])
    print(Points[idx_mid, 0], idx_mid, idx_left, idx_right)'''


    #
    # if Dim == 2:
    #     # 绘制点
    #     plt.scatter(Points[:, 0], Points[:, 1], color='blue')
    #     for i in range(0, Num):
    #         plt.text(Points[i, 0], Points[i, 1], str(i))
    #     # 绘制框架
    #     plt.scatter(Target[0], Target[1], c='red', s=30)
    #     frame_X = np.array([L_bound[0], R_bound[0], R_bound[0], L_bound[0], L_bound[0]])
    #     frame_Y = np.array([L_bound[1], L_bound[1], R_bound[1], R_bound[1], L_bound[1]])
    #     plt.plot(frame_X, frame_Y, color='black')
    #     # 绘制圆
    #     for i in range(0, K):
    #         n = np.linspace(0, 2 * 3.14, 300)
    #         x = can_dis[i] * np.cos(n) + Target[0]
    #         y = can_dis[i] * np.sin(n) + Target[1]
    #         plt.plot(x, y, c='lightsteelblue')
    #         # plt.axis([np.min(L_bound), np.max(R_bound), np.min(L_bound), np.max(R_bound)])
    #     plt.draw()
    #     plt.show()

    '''# 验证正确性
    print('\n---------- Varification of the Correctness----------\n')
    dist_list = np.power(np.sum(np.power(Points - Target, p), 1), 1 / p)
    sorted_dist_list = np.sort(dist_list)
    print('correct_dist_list  : ', np.round(sorted_dist_list[0:K], 4))
    print('sorted_dist_list   : ', np.round(sorted_dist_list, 4))
    print('original_dist_list : ', np.round(dist_list, 4))'''

