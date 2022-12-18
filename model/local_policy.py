import torch
import torch.nn as nn
import torchvision.models as models

from utils.model import get_grid, ChannelPool, Flatten, NNBase


# Local Policy model code
class Local_IL_Policy(NNBase):

    def __init__(self, input_shape, num_actions, recurrent=False,
                 hidden_size=512, deterministic=False):
        '''
        Args:
              input_shape: 3*128*128
              num_actions: 3
              recurrent: 1
              hidden_size: 512
              deterministic: 0
        '''
        super(Local_IL_Policy, self).__init__(recurrent, hidden_size,
                                              hidden_size)

        self.deterministic = deterministic
        self.dropout = 0.5

        resnet = models.resnet18(pretrained=True)
        self.resnet_l5 = nn.Sequential(*list(resnet.children())[0:8])

        # Extra convolution layer
        # 额外加一层卷积层，降低通道数
        self.conv = nn.Sequential(*filter(bool, [
            nn.Conv2d(512, 64, (1, 1), stride=(1, 1)),
            nn.ReLU()
        ]))

        # convolution output size
        input_test = torch.randn(1, 3, input_shape[1], input_shape[2])
        conv_output = self.conv(self.resnet_l5(input_test))
        self.conv_output_size = conv_output.view(-1).size(0)

        # projection layers
        # 后面16维度分别是为了拼接：相对位置的编码，相对角度的编码
        self.proj1 = nn.Linear(self.conv_output_size, hidden_size - 16)
        if self.dropout > 0:
            self.dropout1 = nn.Dropout(self.dropout)
        self.linear = nn.Linear(hidden_size, hidden_size)

        # Short-term goal embedding layers
        self.embedding_angle = nn.Embedding(72, 8)
        # todo 这里为啥是24个bins？
        self.embedding_dist = nn.Embedding(24, 8)

        # Policy linear layer
        self.policy_linear = nn.Linear(hidden_size, num_actions)

        self.train()

    def forward(self, rgb, rnn_hxs, masks, extras):
        '''
        Args:
            rgb: n,3,128,128
            rnn_hxs: n,512
            masks: n,1
            extras: n,2 local_goals # short-term goal
        '''

        if self.deterministic:
            x = torch.zeros(extras.size(0), 3)
            # 这里应该是根据short-term goal
            for i, stg in enumerate(extras):
                if stg[0] < 3 or stg[0] > 68:
                    x[i] = torch.tensor([0.0, 0.0, 1.0])
                elif stg[0] < 36:
                    x[i] = torch.tensor([0.0, 1.0, 0.0])
                else:
                    x[i] = torch.tensor([1.0, 0.0, 0.0])
        else:
            resnet_output = self.resnet_l5(rgb[:, :3, :, :])
            conv_output = self.conv(resnet_output)

            proj1 = nn.ReLU()(self.proj1(conv_output.view(
                -1, self.conv_output_size)))
            if self.dropout > 0:
                proj1 = self.dropout1(proj1)

            angle_emb = self.embedding_angle(extras[:, 0]).view(-1, 8)
            dist_emb = self.embedding_dist(extras[:, 1]).view(-1, 8)

            #
            x = torch.cat((proj1, angle_emb, dist_emb), 1)
            x = nn.ReLU()(self.linear(x))

            #
            if self.is_recurrent:
                # todo _forward_gru？
                x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

            x = nn.Softmax(dim=1)(self.policy_linear(x))

        action = torch.argmax(x, dim=1)

        return action, x, rnn_hxs
