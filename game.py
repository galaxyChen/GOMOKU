# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
""" 

from __future__ import print_function
import numpy as np

class Board(object):
    """
    board for the game
    """

    def __init__(self, **kwargs):
        # 初始化参数
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        self.states = {} #key是move的位置，value是这个move的玩家
        self.n_in_row = int(kwargs.get('n_in_row', 5)) # need how many pieces in a row to win
        self.players = [1, 2] # player1 and player2
        
    def init_board(self, start_player=0):
        # 棋盘不符合规定，报错
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not less than %d' % self.n_in_row)
        # 设定先手的玩家
        self.current_player = self.players[start_player]      
        # 可行解的list
        self.availables = list(range(self.width * self.height))
        # 棋盘状态
        self.states = {} # board states, key:move as location on the board, value:player as pieces type
        # 上一步
        self.last_move = -1

    def move_to_location(self, move):
        """       
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        # 将按顺序展平的棋盘位置编号转变为二维坐标
        h = move  // self.width
        w = move  %  self.width
        return [h, w]

    def location_to_move(self, location):
        # 将二维坐标转变成按顺序展平的一维棋盘
        if(len(location) != 2):
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if(move not in range(self.width * self.height)):
            return -1
        return move

    def current_state(self): 
        """return the board state from the perspective of the current player
        shape: 4*width*height"""
        # 返回当前玩家视角下的棋盘状态
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            # 筛选出当前玩家和对手的行为
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]                       
            # 上一步的我方战局
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            # 上一步的敌方战局
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0 
            # 上一步的落子位置
            square_state[2][self.last_move //self.width, self.last_move % self.height] = 1.0 
        # 先后手的标记
        if len(self.states)%2 == 0:
            square_state[3][:,:] = 1.0
        return square_state[:,::-1,:]

    def do_move(self, move):
        self.states[move] = self.current_player # 设置state，记录下这一步
        self.availables.remove(move) # 从可行解里面去除这一步
        # 切换玩家
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1] 
        # 记录这一个move为上一步
        self.last_move = move

    def has_a_winner(self):
        # 判断是不是有赢家
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row
        # 获得已走的move
        moved = list(set(range(width * height)) - set(self.availables))
        # 没达到最低的move数量要求
        if(len(moved) < self.n_in_row + 2):
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]
            # 检查同一个行
            if (w in range(width - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player
            # 检查同一个列
            if (h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player
            # 检查对角线
            if (w in range(width - n + 1) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player
            if (w in range(n - 1, width) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        # 检查游戏有没有结束
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):#            
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """
    game server
    """

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """
        Draw the board and show game info
        """
        # 画图函数
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')
            
    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """
        start a game between two players
        """
        # 开启两个玩家的一局游戏
        if start_player not in (0,1):
            raise Exception('start_player should be 0 (player1 first) or 1 (player2 first)')
        # 初始化棋盘
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        # 分别设置玩家标识
        player1.set_player_ind(p1)  
        player2.set_player_ind(p2)
        players = {p1: player1, p2:player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while(1):
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player] # 获得player对象
            move = player_in_turn.get_action(self.board) # 传入棋盘，获得下一步的走子
            self.board.do_move(move) # 走子
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner   
            
            
    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree
        store the self-play data: (state, mcts_probs, z)
        """
        # 使用蒙特卡洛玩家进行自我对战
        self.board.init_board()        
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []        
        while(1):
            # 获得当前对局的下一步和对应的概率
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            # 将当前的局面添加到states里面
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # 在棋盘上进行走子
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            # 判断当前有没有游戏结束
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))  
                # 修改每一步的状态评估值为最后的对局结果
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # 重置玩家信息
                player.reset_player() 
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
            