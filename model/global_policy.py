import torch
import torch.nn as nn

from utils.distributions import Categorical, DiagGaussian
from utils.model import get_grid, ChannelPool, Flatten, NNBase


# Global Policy model code
class Global_Policy(NNBase):

    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 downscaling=1):
        '''
        Args:
            obs_shape:8*G*G
            recurrent: args.use_recurrent_global, # 0，全局不使用gru
            hidden_size: g_hidden_size, # 256
            downscaling: args.global_downscaling # 2

        '''
        super(Global_Policy, self).__init__(recurrent, hidden_size,
                                            hidden_size)

        out_size = int(input_shape[1] / 16. * input_shape[2] / 16.)

        # 5 Conv
        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(8, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        # 3 FC
        self.linear1 = nn.Linear(out_size * 32 + 8, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)
        self.critic_linear = nn.Linear(256, 1)

        # orientation embedding
        self.orientation_emb = nn.Embedding(72, 8)
        self.train()

    def forward(self, inputs, rnn_hxs, masks, extras):
        # n,8,G,G -> n,32,G/16,G/16 -> n,7200
        x = self.main(inputs)

        # 将方向为1的离散整型long编码为8长度
        # 离散单位为5°，72*5=360
        # n*8
        orientation_emb = self.orientation_emb(extras).squeeze(1)

        # n*7200+8
        x = torch.cat((x, orientation_emb), 1)

        # n*hidden_size
        x = nn.ReLU()(self.linear1(x))
        if self.is_recurrent:
            # global = False, local = true
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        # n*256
        x = nn.ReLU()(self.linear2(x))

        # n*1
        return self.critic_linear(x).squeeze(-1), x, rnn_hxs

# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L15
class RL_Policy(nn.Module):

    def __init__(self, obs_shape, action_space, model_type=0,
                 base_kwargs=None):
        '''
        Args:
            obs_shape:8*G*G
            action_space: gym.spaces.Box(low=0.0, high=1.0,shape=(2,), dtype=np.float32)
            model_type = 固定为0，因为是从上面github链接上迁移过来的，原本是有很多模型类别，这里只需要全局
            base_kwargs={'recurrent': args.use_recurrent_global, # 0，全局不使用gru
                         'hidden_size': g_hidden_size, # 256
                         'downscaling': args.global_downscaling # 2
                        }).to(device)
        '''
        super(RL_Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if model_type == 0:
            self.network = Global_Policy(obs_shape, **base_kwargs)
        else:
            raise NotImplementedError

        # 传进来的gym.spaces
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n # 空间所有动作
            self.dist = Categorical(self.network.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0] # 2

            # todo 这里的作用？
            # 初始化高斯分布
            self.dist = DiagGaussian(self.network.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.model_type = model_type

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return self.network.rec_state_size

    def forward(self, inputs, rnn_hxs, masks, extras):
        if extras is None:
            return self.network(inputs, rnn_hxs, masks)
        else:
            return self.network(inputs, rnn_hxs, masks, extras)

    def act(self, inputs, rnn_hxs, masks, extras=None, deterministic=False):

        # 奖励值n,1
        # 经过relu n,256
        # 没用rnn_hxs = torch.zeros((num_steps + 1, num_processes, extras_size),dtype=torch.long)
        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)

        # 根据action初始化分布
        dist = self.dist(actor_features)

        if deterministic: # 确定的就返回均值，即直接经过一层fc的值
            action = dist.mode()
        else: # 随机的就根据概率进行采样
            action = dist.sample()

        # 返回每个action的概率
        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, extras=None):
        value, _, _ = self(inputs, rnn_hxs, masks, extras)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, extras=None):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
