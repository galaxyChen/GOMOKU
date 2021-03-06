# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch (tested in PyTorch 0.2.0 and 0.3.0)

@author: Junxiao Song
""" 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        # 初始化棋盘大小
        self.board_width = board_width
        self.board_height = board_height
        # 定义三个不同大小的卷积层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 行为策略价值网络的定义，一个卷积层加上一个线性层
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height, board_width*board_height)
        # 局面评估函数的定义，一个卷积层加上两个线性层
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)
    
    def forward(self, state_input):
        # 连续进行三个卷积操作
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 行为价值网络的计算，对上面卷积操作的结果在进行卷积
        x_act = F.relu(self.act_conv1(x))
        # 将上一步的卷积结果展平成一维向量，第一个维度是batch_size
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        # 对一维向量进行log softmax操作
        x_act = F.log_softmax(self.act_fc1(x_act),dim=1)  
        # 状态价值网络
        x_val = F.relu(self.val_conv1(x)) # 卷积
        x_val = x_val.view(-1, 2*self.board_width*self.board_height) # 展平
        x_val = F.relu(self.val_fc1(x_val)) # 全连接
        x_val = F.tanh(self.val_fc2(x_val))# 全连接输出评估值
        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height, net_params=None, use_gpu=True):        
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # l2 正则项的参数
        # 加载价值策略网咯
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()       
        else:
            self.policy_value_net = Net(board_width, board_height)
        # 使用Adam算法进行优化
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        if net_params:
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values 
        """
        # 输入为当前局面
        # 输出为当前局面的概率和场面评估值
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch) # 从策略价值网络中得到当前局面的走子概率和值
            act_probs = np.exp(log_act_probs.data.cpu().numpy()) # 将log概率恢复成原本的softmax概率
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())            
            return act_probs, value.data.numpy()
        

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        # 输入一个棋盘
        # 输出一个(行为，概率)的list，每一个行为都是当前有效的走子
        # 同时会输出场面分数
        legal_positions = board.availables #获得可行解
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 4, self.board_width, self.board_height)) # 获得当前的局面
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).cuda().float()) # 将当前局面输入价值网络得到走子概率和局面估计
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten()) # 从log softmax还原softmax概率
        else:
            log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).float()) 
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions]) # 筛选出可行解和对应的概率
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # 包装变量，格式转换
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # 清空累计的梯度
        self.optimizer.zero_grad()
        # 设置学习率
        set_learning_rate(self.optimizer, lr)

        # 使用价值网络算出走子概率和局面评估值
        log_act_probs, value = self.policy_value_net(state_batch)
        # loss = (z - v)^2 - pi^T * log(p) + c||theta||^2 （优化算法自带L2正则项）
        value_loss = F.mse_loss(value.view(-1), winner_batch) # 局面评估值的loss，将局面评估值（-1,1）和真实对局结果（1或者-1）进行作差平方得到loss
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1)) # 走子的loss，使用交叉熵
        loss = value_loss + policy_loss
        # 进行后向传播
        loss.backward()
        self.optimizer.step()
        # 计算策略的熵，仅用于监控
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.data[0], entropy.data[0]

    def get_policy_param(self):
        # 返回策略价值网络的参数
        net_params = self.policy_value_net.state_dict()
        return net_params
