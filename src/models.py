"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""
'''更多的优化方向：
进一步利用相机内参，提升模型性能： 【内参位置编码】
量产车辆不同外参条件下模型性能的退化： 【外参扰动】
单帧网络模型的性能瓶颈： 【引入时序信息】
Voxel_pooling部署难： 【voxel_pooling替代方案】
'''

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)  # 上采样 BxCxHxW->BxCx2Hx2W

        self.conv = nn.Sequential(  # 两个3x3卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # inplace=True使用原地操作，节省内存
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)  # 对x1进行上采样
        x1 = torch.cat([x2, x1], dim=1)  # 将x1和x2 concat 在一起
        return self.conv(x1)


class CamEncode(nn.Module):  # 提取图像特征进行图像编码
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D  # 41
        self.C = C  # 64

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")  # 使用 efficientnet 提取特征

        self.up1 = Up(320+112, 512)  # 上采样模块，输入输出通道分别为320+112和512
        # 从512维度conv到了D+C的维度!!
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)  # 1x1卷积，变换维度

    def get_depth_dist(self, x, eps=1e-20):  # 对深度维进行softmax，得到每个像素不同深度的概率
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)  # 使用efficientnet提取特征  x: 24 x 512 x 8 x 22
        # Depth
        x = self.depthnet(x)  # 1x1卷积变换维度  x: 24 x 105(C+D) x 8 x 22
        # B x (D + C)
        depth = self.get_depth_dist(x[:, :self.D])  # 第二个维度的前D个作为深度维，进行softmax  depth: 24 x 41 x 8 x 22
        # 骚操作，把D和C又从一个维度上拆成两个维度，前面为了加速计算
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)  # 将特征通道维和通道维利用广播机制相乘  new_x: 24 x 64 x 41 x 8 x 22

        return depth, new_x

    def get_eff_depth(self, x):  # 使用efficientnet提取特征
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))  #  x: 24 x 32 x 64 x 176
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x  # x: 24 x 320 x 4 x 11
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])  # 先对endpoints[4]进行上采样，然后将 endpoints[5]和endpoints[4] concat 在一起
        return x  # x: 24 x 512 x 8 x 22

    def forward(self, x):
        depth, x = self.get_depth_feat(x)  # depth: B*N x D x fH x fW(24 x 41 x 8 x 22)  x: B*N x C x D x fH x fW(24 x 64 x 41 x 8 x 22)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):  # inC: 64  outC: 1
        super(BevEncode, self).__init__()

        # 使用resnet的前3个stage作为backbone
        # 大卷积把深度投放不准的信息抹回来
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(  # 2倍上采样->3x3卷积->1x1卷积
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):  # x: 4 x 64 x 200 x 200
        x = self.conv1(x)  # x: 4 x 64 x 100 x 100
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)  # x1: 4 x 64 x 100 x 100
        x = self.layer2(x1)  # x: 4 x 128 x 50 x 50
        x = self.layer3(x)  # x: 4 x 256 x 25 x 25

        x = self.up1(x, x1)  # 给x进行4倍上采样然后和x1 concat 在一起  x: 4 x 256 x 100 x 100
        x = self.up2(x)  # 2倍上采样->3x3卷积->1x1卷积  x: 4 x 1 x 200 x 200

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf   # 网格配置参数
        self.data_aug_conf = data_aug_conf   # 数据增强配置参数

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )  # 划分网格
        self.dx = nn.Parameter(dx, requires_grad=False)  # [0.5,0.5,20],xyz的分辨率
        self.bx = nn.Parameter(bx, requires_grad=False)  # [-49.5,-49.5,0],xyz的起始值
        self.nx = nn.Parameter(nx, requires_grad=False)  # [200,200,1],xyz的格子数量

        self.downsample = 16  # 下采样倍数
        self.camC = 64  # 图像特征维度
        self.frustum = self.create_frustum()  # frustum: DxfHxfWx3(41x8x22x3)
        self.D, _, _, _ = self.frustum.shape  # D: 41
        self.camencode = CamEncode(self.D, self.camC, self.downsample)  # backbone
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    def create_frustum(self):# 为每一张图片生成一个棱台状（frustum）的点云
        # make grid in image plane
        # 数据增强后图片大小  ogfH:128  ogfW:352
        ogfH, ogfW = self.data_aug_conf['final_dim']  # 原始图片大小  ogfH:128  ogfW:352
        # 下采样16倍后图像大小  fH: 128/16=8  fW: 352/16=22
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  # 下采样16倍后图像大小  fH: 8  fW: 22
        '''
        ds： 在深度方向上划分网格 
        dbound: [4.0, 45.0, 1.0]  
        arange后-> [4.0,5.0,6.0,...,44.0]
        view后(相当于reshape操作)-> (41x1x1)    
        expand后（扩展张量中某维数据的尺寸）->  ds: DxfHxfW(41x8x22)
        这行代码使用 PyTorch 创建一个 3D 张量 ds，其维度为 (D, fH, fW)，其中 D 表示深度方向上的网格数量，fH 和 fW 分别表示图像的高度和宽度。具体而言，这行代码使用 torch.arange 函数生成一个从 self.grid_conf['dbound'] 中指定的起始点、终止点和步长构成的等差数列，并将其转换为浮点型。然后，使用 view 函数将该等差数列重塑为列向量形式，并在深度方向上进行复制以匹配图像的高度和宽度，最终得到一个 DxfHxfW 的 3D 张量 ds，表示在深度方向上划分的网格。
        view为reshape操作，-1为自动计算，后两个为1，计算得到结果为41x1x1
        再expand在后两个维度上复制扩展至fH,fW，即41x8x22
        最后arrange将其填充为线性4.0, 5.0, 6.0 ... 44.0，代表深度距离编码
        '''
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)  # 在深度方向上划分网格 ds: DxfHxfW(41x8x22) 
        #  _ 作为占位符，表示在这里并不需要使用到这个变量，只是为了占位而已。
        D, _, _ = ds.shape # D: 41 表示深度方向上网格的数量
        '''
        xs: 在宽度方向上划分网格
        linspace 后(在[0,ogfW)区间内，均匀划分fW份)-> [0,16,32..336]  大小=fW(22)   
        view后-> 1x1xfW(1x1x22)
        expand后-> xs: DxfHxfW(41x8x22)
        '''
        # ???需要debug实际跑一下
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)  # 在0到351上划分22个格子 xs: DxfHxfW(41x8x22)
        '''
        ys: 在高度方向上划分网格
        linspace 后(在[0,ogfH)(0,128)区间内，均匀划分fH份)-> [0,16,32..112]  大小=fH(8)
        view 后-> 1xfHx1 (1x8x1)
        expand 后-> ys: DxfHxfW (41x8x22)
        把下采样后尺度上，对应原图中y上坐标值放到网格值里
        '''
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)  # 在0到127上划分8个格子 ys: DxfHxfW(41x8x22)

        # D x H x W x 3,最后一维跟原图做了映射 ???debug
        frustum = torch.stack((xs, ys, ds), -1)  # 堆积起来形成网格坐标, frustum[i,j,k,0]就是(i,j)位置，深度为k的像素的宽度方向上的栅格坐标   frustum: DxfHxfWx3
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape  # B:4(batchsize)    N: 6(相机数目)

        # undo post-transformation 增广
        # B x N x D x H x W x 3
        # 抵消数据增强及预处理对像素的变化(post_trans / post_rots)旋转平移
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego [[u,v] * Zc, Zc]（相机内外参公式）,其中Zc=d
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)  # 将像素坐标(u,v,d)变成齐次坐标(du,dv,d)
        # d[u,v,1]^T=intrins*rots^(-1)*([x,y,z]^T-trans)
        # 此处的rots为相机到车身，所以不需要求逆
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)  # 将像素坐标d[u,v,1]^T转换到车体坐标系下的[x,y,z]^T

        return points  # B x N x D x H x W x 3 (4 x 6 x 41 x 8 x 22 x 3)

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape  # B: 4  N: 6  C: 3  imH: 128  imW: 352

        x = x.view(B*N, C, imH, imW)  # B和N两个维度合起来  x: 24 x 3 x 128 x 352
        x = self.camencode(x) # 进行图像编码  x: B*N x C x D x fH x fW(24 x 64 x 41 x 8 x 22)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)  #将前两维拆开 x: B x N x C x D x fH x fW(4 x 6 x 64 x 41 x 8 x 22)
        x = x.permute(0, 1, 3, 4, 5, 2)  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 8 x 22 x 64)

        return x

    def voxel_pooling(self, geom_feats, x):
        '''
        在投射过程中会存在的3点问题：
        1、当下为车身坐标系，我们希望得到bev图像feature坐标系;
        2、存在一些范围外的点需要滤掉;
        3、如何处理落在同一个bev各自内的点;
        '''
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 8 x 22 x 3)
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 8 x 22 x 64)

        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 8  W: 22  C: 64
        Nprime = B*N*D*H*W  # Nprime: 173184

        # flatten x
        x = x.reshape(Nprime, C)  # 将图像展平，一共有 B*N*D*H*W 个点

        # flatten indices  bx：三个方向上的起始值(x,y,d)，dx：三个方向上的分辨率(一格子多少米)。???debug
        # geom_feats - (50 - 1/2)
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()  # 将[-50,50] [-10 10]的范围平移到[0,100] [0,20]，计算栅格坐标并取整，把位置拉到正值范围
        geom_feats = geom_feats.view(Nprime, 3)  # 将像素映射关系同样展平  geom_feats: B*N*D*H*W x 3 (173184 x 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])  # 每个点对应于哪个batch
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # geom_feats: B*N*D*H*W x 4(173184 x 4), geom_feats[:,3]表示batch_id

        # filter out points that are outside box
        # 过滤掉在边界线之外的点 x:0~199  y: 0~199  z: 0
        # 第一行：x >= 0 & x < 200，nx=(200,200,1)格子的尺寸,保留范围内，过滤掉范围外的点
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]  # x: 168648 x 64
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]  # 给每一个点一个rank值(index)，rank相等的点在同一个batch，并且在在同一个格子里面
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 按照rank排序，这样rank相近的点就在一起了
        # x: 168648 x 64  geom_feats: 168648 x 4  ranks: 168648

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)  # 一个batch的一个格子里只留一个点 x: 29072 x 64  geom_feats: 29072 x 4

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x 1 x 200 x 200
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)  # 消除掉z维

        return final  # final: 4 x 64 x 200 x 200

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)  # 像素坐标到自车中坐标的映射关系 geom: B x N x D x H x W x 3 (4 x 6 x 41 x 8 x 22 x 3)
        x = self.get_cam_feats(x)  # 提取图像特征并预测深度编码 x: B x N x D x fH x fW x C(4 x 6 x 41 x 8 x 22 x 64)

        x = self.voxel_pooling(geom, x)  # x: 4 x 64 x 200 x 200

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        # x:[4,6,3,128,352]
        # rots: [4,6,3,3]
        # trans: [4,6,3]
        # intrins: [4,6,3,3]
        # post_rots: [4,6,3,3]
        # post_trans: [4,6,3]
                
        #  x:[4,6,3,128,352] 输入图像
        # rots：由相机坐标系->车身坐标系的旋转矩阵，rots = (bs, N, 3, 3)；
        #
        # trans：由相机坐标系->车身坐标系的平移矩阵，trans=(bs, N, 3)；
        #
        # intrinsic：相机内参，intrinsic = (bs, N, 3, 3)；
        #
        # post_rots：由图像增强引起的旋转矩阵，post_rots = (bs, N, 3, 3)；
        #
        # post_trans：由图像增强引起的平移矩阵，post_trans = (bs, N, 3)；

        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)  # 将图像转换到BEV下，x: B x C x 200 x 200 (4 x 64 x 200 x 200)
        x = self.bevencode(x)  # 用resnet18提取特征  x: 4 x 1 x 200 x 200
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
