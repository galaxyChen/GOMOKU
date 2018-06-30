# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value network
to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""
import numpy as np
import copy 


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """
    # 蒙特卡洛树的节点类

    def __init__(self, parent, prior_p):
        self._parent = parent # 父节点
        self._children = {}  # 子节点集合
        self._n_visits = 0 # 访问次数
        self._Q = 0 # 节点平均行为价值Q值
        self._u = 0 # U值
        self._P = prior_p # 被选择的先验概率P值，初始化为神经网络的输出

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors -- output from policy function - a list of tuples of actions
            and their prior probability according to the policy function.
        """
        # 使用策略网络输出的走子概率作为初始的先验概率，将每一个行为作为子节点进行扩展
        for action, prob in action_priors:
            # 如果这个行为不是当前节点的子节点，那么添加一个新的子节点
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value, Q plus bonus u(P).
        Returns:
        A tuple of (action, next_node)
        """
        # 选择P+U值最大的节点
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        Arguments:
        leaf_value -- the value of subtree evaluation from the current player's perspective.        
        """
        # 访问次数加1
        self._n_visits += 1
        # 更新Q值
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # 如果当前节点不是根节点，则递归调用父节点的节点更新函数
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        # 父节点已经更新完毕，更新当前节点
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        """
        # 利用给定常数计算Q+U值
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search.
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """Arguments:
        policy_value_fn -- a function that takes in a board state and outputs a list of (action, probability)
            tuples and also a score in [-1, 1] (i.e. the expected value of the end game score from 
            the current player's perspective) for the current player.
        c_puct -- a number in (0, inf) that controls how quickly exploration converges to the
            maximum-value policy, where a higher value means relying on the prior more
        """
        # policy_value_fn：一个根据当前对局情况返回一个可行解和可行解的概率的函数
        # policy_value_fn同时返回一个[-1,1]的实数值，代表当前的局面评估值
        # c_puct一个控制搜索收敛速度的正实数，一个大的c_puct值表示更加依赖于先验概率
        self._root = TreeNode(None, 1.0) #建树
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at the leaf and
        propagating it back through its parents. State is modified in-place, so a copy must be
        provided.
        Arguments:
        state -- a copy of the state.
        """
        # 从根到叶子进行一次简单的对局，得到叶子的值并将其反向传播到父节点
        node = self._root
        while(1):
            # 如果当前节点是一个叶子结点，则退出对局
            if node.is_leaf():
                break                
            # 选择一个P+U值最大的节点进行走子
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # 根据神经网络输出的（行为，概率）列表对叶子结点进行评估
        # 神经网络会输出当前玩家的走子概率和局面评估值
        action_probs, leaf_value = self._policy(state)
        # 检查当前对局是否结束
        end, winner = state.game_end()
        if not end:
            # 如果没有结束，则根据可行解扩展子节点
            node.expand(action_probs)
        else:
            # 对局结束，将节点的值设置为真实的对局评估，1或者-1
            if winner == -1:  # 平局
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0

        # 更新节点的值
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Runs all playouts sequentially and returns the available actions and their corresponding probabilities 
        Arguments:
        state -- the current state, including both game state and the current player.
        temp -- temperature parameter in (0, 1] that controls the level of exploration
        Returns:
        the available actions and the corresponding probabilities 
        """
        # 顺序返回所有对局和可行解还有对应的概率
        # state：对局的状态
        # temp：温度参数，控制搜索的程度
        # 返回可行解和对应的概率
        for n in range(self._n_playout):#对每一次对局进行操作：
            state_copy = copy.deepcopy(state)#深拷贝一个对局状态，以防后续操作影响原来的变量内容
            self._playout(state_copy)#进行一次对局

        # 基于访问次数计算走子概率
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))       
         
        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know about the subtree.
        """
        if last_move in self._root._children:
            # 如果上一步是当前节点的子节点（即上一步的走法在模拟中出现过）
            self._root = self._root._children[last_move] # 将对应的子节点设为根节点
            self._root._parent = None
        else:
            # 否则新建一个根节点
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"
        

class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        # 初始化一个蒙特卡洛树
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        # 是否为自我对局
        self._is_selfplay = is_selfplay
    
    def set_player_ind(self, p):
        # 设置玩家
        self.player = p

    def reset_player(self):
        # 重置玩家
        self.mcts.update_with_move(-1) # 该调用会使得蒙特卡洛树重置为空

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables # 获得当前对局的可行走子
        move_probs = np.zeros(board.width*board.height) # 初始化所有走子概率为0
        if len(sensible_moves) > 0:
            # 如果存在可行的走子方案
            # 将温度参数和对局传进蒙特卡洛树进行对局搜索，获得每一个行为的概率
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs #设置行为概率向量pi
            if self._is_selfplay:
                # 在自我对战中添加Dirichlet噪音，添加对局的不确定性
                # 在走子概率中加上Dirichlet噪音，并按照这个走子概率走下一步棋
                move = np.random.choice(acts, p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
                # 将目前的走子更新到蒙特卡洛树
                self.mcts.update_with_move(move) #
            else:
                # 非自我对战的情况，根据概率选择走子。在temp=1e-3的情况下，非常接近于选择最大概率的走子
                move = np.random.choice(acts, p=probs)       
                # 重置当前的蒙特卡洛树
                self.mcts.update_with_move(-1)             
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))
            
            # 判断是否需要返回走子概率
            if return_prob:
                return move, move_probs
            else:
                return move
        else:    
            # 没有可行走子了，输出警告        
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)    