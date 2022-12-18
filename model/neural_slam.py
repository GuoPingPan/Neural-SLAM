import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

from utils.model import get_grid, ChannelPool, Flatten, NNBase

# Neural SLAM Module code
class Neural_SLAM_Module(nn.Module):
    """
    """

    def __init__(self, args):
        super(Neural_SLAM_Module, self).__init__()

        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.use_pe = args.use_pose_estimation

        # Visual Encoding
        resnet = models.resnet18(pretrained=args.pretrained_resnet)
        # 取resnet18的前8个子模块，去掉了avgpool和fc
        # *是元组，**是字典
        # 降采样32
        self.resnet_l5 = nn.Sequential(*list(resnet.children())[0:8])
        self.conv = nn.Sequential(*filter(bool, [
            nn.Conv2d(512, 64, (1, 1), stride=(1, 1)),
            nn.ReLU()
        ]))

        # convolution output size
        # 输入图像大小1*3*128*128
        input_test = torch.randn(1,
                                 self.n_channels,
                                 self.screen_h,
                                 self.screen_w)
        conv_output = self.conv(self.resnet_l5(input_test))

        self.pool = ChannelPool(1) # 对Channel做pooling
        # output展平
        self.conv_output_size = conv_output.view(-1).size(0)

        # projection layer
        self.proj1 = nn.Linear(self.conv_output_size, 1024)
        self.proj2 = nn.Linear(1024, 4096)

        if self.dropout > 0:
            self.dropout1 = nn.Dropout(self.dropout)
            self.dropout2 = nn.Dropout(self.dropout)

        # Deconv layers to predict map
        self.deconv = nn.Sequential(*filter(bool, [
            nn.ConvTranspose2d(64, 32, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 2, (4, 4), stride=(2, 2), padding=(1, 1)),
        ]))

        # Pose Estimator
        # 地图是2维的，t-1时刻和t时刻总共4维
        self.pose_conv = nn.Sequential(*filter(bool, [
            nn.Conv2d(4, 64, (4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 32, (4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 16, (3, 3), stride=(1, 1)),
            nn.ReLU()
        ]))


        pose_conv_output = self.pose_conv(torch.randn(1, 4,
                                                      self.vision_range,
                                                      self.vision_range))

        # 将输出展平
        self.pose_conv_output_size = pose_conv_output.view(-1).size(0)

        # projection layer
        # 估计相对两帧梯度的相对位姿
        self.pose_proj1 = nn.Linear(self.pose_conv_output_size, 1024)
        self.pose_proj2_x = nn.Linear(1024, 128)
        self.pose_proj2_y = nn.Linear(1024, 128)
        self.pose_proj2_o = nn.Linear(1024, 128)
        self.pose_proj3_x = nn.Linear(128, 1)
        self.pose_proj3_y = nn.Linear(128, 1)
        self.pose_proj3_o = nn.Linear(128, 1)



        if self.dropout > 0:
            self.pose_dropout1 = nn.Dropout(self.dropout)

        self.st_poses_eval = torch.zeros(args.num_processes,
                                         3).to(self.device)
        self.st_poses_train = torch.zeros(args.slam_batch_size,
                                          3).to(self.device)

        grid_size = self.vision_range * 2
        self.grid_map_eval = torch.zeros(args.num_processes, 2,
                                         grid_size, grid_size
                                         ).float().to(self.device)
        self.grid_map_train = torch.zeros(args.slam_batch_size, 2,
                                          grid_size, grid_size
                                          ).float().to(self.device)

        self.agent_view = torch.zeros(args.num_processes, 2,
                                      self.map_size_cm // self.resolution,
                                      self.map_size_cm // self.resolution
                                      ).float().to(self.device)

    def forward(self, obs_last, obs, poses, maps, explored, current_poses,
            build_maps=True):

        # Get egocentric map prediction for the current obs
        bs, c, h, w = obs.size()
        # n,512,128/32=4,4
        resnet_output = self.resnet_l5(obs[:, :3, :, :])

        # n,64,128/32=4,4
        conv_output = self.conv(resnet_output)

        # n*1024
        proj1 = nn.ReLU()(self.proj1(
                          conv_output.view(-1, self.conv_output_size)))
        if self.dropout > 0:
            proj1 = self.dropout1(proj1)

        # n*4096
        proj3 = nn.ReLU()(self.proj2(proj1))

        deconv_input = proj3.view(bs, 64, 8, 8)

        # n,2,64,64
        # 2,v,v
        deconv_output = self.deconv(deconv_input)

        # sigmoid限制在0到1之间
        pred = torch.sigmoid(deconv_output)

        # 占用概率和被探索的概率
        proj_pred = pred[:, :1, :, :]
        fp_exp_pred = pred[:, 1:, :, :]

        with torch.no_grad():
            # Get egocentric map prediction for the last obs
            bs, c, h, w = obs_last.size()
            resnet_output = self.resnet_l5(obs_last[:, :3, :, :])
            conv_output = self.conv(resnet_output)

            proj1 = nn.ReLU()(self.proj1(
                              conv_output.view(-1, self.conv_output_size)))
            if self.dropout > 0:
                proj1 = self.dropout1(proj1)
            proj3 = nn.ReLU()(self.proj2(proj1))

            deconv_input = proj3.view(bs, 64, 8, 8)
            deconv_output = self.deconv(deconv_input)
            pred_last = torch.sigmoid(deconv_output)

            # ST of proj
            vr = self.vision_range
            # todo 为啥要扩大两倍
            # 应该是为了避免仿射变换超出边界
            grid_size = vr * 2

            if build_maps:
                # detach可以让st_poses和计算图断开
                st_poses = self.st_poses_eval.detach_()
                grid_map = self.grid_map_eval.detach_()
            else:
                st_poses = self.st_poses_train.detach_()
                grid_map = self.grid_map_train.detach_()

            st_poses.fill_(0.)
            # 似乎是相对位姿
            # pose应该是n*3
            # todo 这里还搞不清楚传回来的pose是什么？按照论文应该是mt-1和mt之间的位姿变换
            st_poses[:, 0] = poses[:, 1] * 200. / self.resolution / grid_size
            st_poses[:, 1] = poses[:, 0] * 200. / self.resolution / grid_size
            st_poses[:, 2] = poses[:, 2] * 57.29577951308232

            #
            rot_mat, trans_mat = get_grid(st_poses,
                                          (bs, 2, grid_size, grid_size),
                                          self.device)

            # n,2,2*vr,2*vr
            grid_map.fill_(0.)
            # todo 这里为什么纵轴不放在中间而放在下半区域呢？
            grid_map[:, :, vr:, int(vr / 2):int(vr / 2 + vr)] = pred_last
            # https://blog.csdn.net/Epoch511/article/details/118155676
            translated = F.grid_sample(grid_map, trans_mat)
            rotated = F.grid_sample(translated, rot_mat)
            rotated = rotated[:, :, vr:, int(vr / 2):int(vr / 2 + vr)]

            pred_last_st = rotated

        # Pose estimator
        # 4*64*64
        pose_est_input = torch.cat((pred.detach(), pred_last_st.detach()),
                                   dim=1)

        # PE 3Conv+3FC
        pose_conv_output = self.pose_conv(pose_est_input)
        pose_conv_output = pose_conv_output.view(-1,
                                                 self.pose_conv_output_size)

        proj1 = nn.ReLU()(self.pose_proj1(pose_conv_output))

        if self.dropout > 0:
            proj1 = self.pose_dropout1(proj1)

        proj2_x = nn.ReLU()(self.pose_proj2_x(proj1))
        pred_dx = self.pose_proj3_x(proj2_x)

        proj2_y = nn.ReLU()(self.pose_proj2_y(proj1))
        pred_dy = self.pose_proj3_y(proj2_y)

        proj2_o = nn.ReLU()(self.pose_proj2_o(proj1))
        pred_do = self.pose_proj3_o(proj2_o)

        pose_pred = torch.cat((pred_dx, pred_dy, pred_do), dim=1)

        # 是否采用位置更新，不采用则置为0
        if self.use_pe == 0:
            pose_pred = pose_pred * self.use_pe

        if build_maps:
            # Aggregate egocentric map prediction in the geocentric map
            # using the predicted pose
            with torch.no_grad():
                #  self.map_size_cm // self.resolution = 480
                agent_view = self.agent_view.detach_()
                agent_view.fill_(0.)

                # 地图中心减去vr
                x1 = self.map_size_cm // (self.resolution * 2) \
                        - self.vision_range // 2
                x2 = x1 + self.vision_range
                y1 = self.map_size_cm // (self.resolution * 2)
                y2 = y1 + self.vision_range
                agent_view[:, :, y1:y2, x1:x2] = pred

                corrected_pose = poses + pose_pred

                def get_new_pose_batch(pose, rel_pose_change):
                    pose[:, 1] += rel_pose_change[:, 0] * \
                                  torch.sin(pose[:, 2] / 57.29577951308232) \
                                  + rel_pose_change[:, 1] * \
                                  torch.cos(pose[:, 2] / 57.29577951308232)
                    pose[:, 0] += rel_pose_change[:, 0] * \
                                  torch.cos(pose[:, 2] / 57.29577951308232) \
                                  - rel_pose_change[:, 1] * \
                                  torch.sin(pose[:, 2] / 57.29577951308232)
                    pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

                    pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
                    pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

                    return pose

                # current_poses=local_pose，并且local_pose是迭代的，local_pose是在local_map的表达
                current_poses = get_new_pose_batch(current_poses,
                                                   corrected_pose)
                st_pose = current_poses.clone().detach()

                #                           离散化                                         局部地图大小                                  局部地图大小
                st_pose[:, :2] = - (st_pose[:, :2] * 100.0 / self.resolution - self.map_size_cm // (self.resolution * 2)) / (self.map_size_cm // (self.resolution * 2))
                st_pose[:, 2] = 90. - (st_pose[:, 2])

                rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                              self.device)

                rotated = F.grid_sample(agent_view, rot_mat)
                translated = F.grid_sample(rotated, trans_mat)


                # maps=local_map，将转以后的map和local_map进行通道维度的cat，然后进行maxpooling更新地图
                # todo 这里依照的是两者的最大值，而不是两者的叠加
                maps2 = torch.cat((maps.unsqueeze(1),
                                   translated[:, :1, :, :]), 1)
                explored2 = torch.cat((explored.unsqueeze(1),
                                       translated[:, 1:, :, :]), 1)

                map_pred = self.pool(maps2).squeeze(1)
                exp_pred = self.pool(explored2).squeeze(1)

        else:
            map_pred = None
            exp_pred = None
            current_poses = None

        return proj_pred, fp_exp_pred, map_pred, exp_pred, pose_pred, current_poses
