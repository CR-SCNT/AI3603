"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板（启发式搜索）
- MCTSAgent: 蒙特卡洛树搜索实现（新增，第535行）
- MCTSNode: MCTS树节点数据结构（新增，第460行）
- analyze_shot_for_reward: 击球结果评分函数

MCTS相关文件：
- MCTS_GUIDE.md: 详细实现说明和原理
- MCTS_CHECKLIST.md: 实施步骤清单
- MCTS_QUICKREF.md: 快速参考卡
- test_mcts.py: MCTS框架测试脚本
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
from enum import Enum
import copy
import os
from datetime import datetime
import random
import signal
from threading import Timer
import threading
import platform
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟
    
    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回：
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明：
        在 Unix/Linux 上使用 signal.SIGALRM，在 Windows 上使用线程超时
        超时后自动恢复，不会导致程序卡死
    """
    is_unix = platform.system() in ('Linux', 'Darwin')
    
    if is_unix:
        # Unix/Linux 使用 signal.SIGALRM
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)  # 设置超时时间
        
        try:
            pt.simulate(shot, inplace=True)
            signal.alarm(0)  # 取消超时
            return True
        except SimulationTimeoutError:
            print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
            return False
        except Exception as e:
            signal.alarm(0)  # 取消超时
            raise e
        finally:
            signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器
    else:
        # Windows 使用线程超时机制
        result = [False]
        exception = [None]
        
        def simulate_task():
            try:
                pt.simulate(shot, inplace=True)
                result[0] = True
            except Exception as e:
                exception[0] = e
        
        thread = Timer(timeout, lambda: None)
        thread.daemon = True
        
        import threading
        sim_thread = threading.Thread(target=simulate_task, daemon=True)
        sim_thread.start()
        sim_thread.join(timeout=timeout)
        
        if sim_thread.is_alive():
            print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
            return False
        
        if exception[0] is not None:
            raise exception[0]
        
        return result[0]

# ============================================



def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 4. 计算奖励分数
    score = 0
    
    # 白球进袋处理
    if cue_pocketed and eight_pocketed:
        score -= 150  # 白球+黑8同时进袋，严重犯规
    elif cue_pocketed:
        score -= 100  # 白球进袋
    elif eight_pocketed:
        # 黑8进袋：只有清台后（player_targets == ['8']）才合法
        if player_targets == ['8']:
            score += 100  # 合法打进黑8
        else:
            score -= 150  # 清台前误打黑8，判负
            
    # 首球犯规和碰库犯规
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    # 进球得分（own_pocketed 已根据 player_targets 正确分类）
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    # 合法无进球小奖励
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10  
        
    return score

class Agent():
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action



class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""
    
    def __init__(self, target_balls=None):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()
        
        # 搜索空间
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        # 优化参数
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2
        
        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        print("BasicAgent (Smart, pooltool-native) 已初始化。")

    
    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        
        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子
        
        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer


    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            print(f"[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建“奖励函数” (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                        phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                        theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                        a_noisy = a + np.random.normal(0, self.noise_std['a'])
                        b_noisy = b + np.random.normal(0, self.noise_std['b'])
                        
                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)
                        
                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    
                    # 关键：使用带超时保护的物理模拟（3秒上限）
                    if not simulate_with_timeout(shot, timeout=3):
                        return 0  # 超时是物理引擎问题，不惩罚agent
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的“裁判”来打分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )


                return score

            print(f"[BasicAgent] 正在为 Player (targets: {my_targets}) 搜索最佳击球...")
            
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )
            
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']

            if best_score < 10:
                print(f"[BasicAgent] 未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。")
                return self._random_action()
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            print(f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action

        except Exception as e:
            print(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()

class NewAgent(Agent):
    """基于集合启发搜索的自定义 Agent"""
    
    def __init__(self):
        pass
    
    def decision(self, balls=None, my_targets=None, table=None):
        """决策方法
        
        参数：
            observation: (balls, my_targets, table)
        
        返回：
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        from utils import vec2, norm, angle_deg, seg_dist
        if balls is None or my_targets is None or table is None:
            print("[NewAgent] 决策时缺少必要参数，使用随机动作。")
            return self._random_action()
        try:
            remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining) == 0:
                my_targets = ["8"]
            R = 0.028575 # 球半径常量
            cue_pos = vec2(balls["cue"].state.rvw[0])
            pockets = list(table.pockets.values())
            candidates = []
            
            # 搜索所有目标球和袋口的组合，生成候选击球方案
            for bid in my_targets:
                if balls[bid].state.s == 4:
                    continue
                tgt = vec2(balls[bid].state.rvw[0])
                for pk in pockets:
                    pkc = vec2(pk.center)
                    dir_tp = norm(pkc - tgt)
                    ghost = tgt - 2.0 * R * dir_tp
                    if np.linalg.norm(cue_pos - ghost) < 1e-6:
                        continue
                    # 检查到ghost路径是否被其他球阻挡
                    blocked = False
                    for ball_id, obj in balls.items():
                        if ball_id in ["cue"]:
                            continue
                        if ball_id == bid:
                            continue
                        if obj.state.s == 4:
                            continue
                        ball_pos = vec2(obj.state.rvw[0])
                        if seg_dist(cue_pos, ghost, ball_pos) < 2.0 * R - 1e-3:
                            blocked = True
                            break
                    if blocked:
                        continue
                    # 检查tgt到pkc路径是否被其他球阻挡
                    blocked2 = False
                    for ball_id, obj in balls.items():
                        if ball_id in ["cue", bid]:
                            continue
                        if obj.state.s == 4:
                            continue
                        ball_pos = vec2(obj.state.rvw[0])
                        if seg_dist(tgt, pkc, ball_pos) < 2.0 * R - 1e-3:
                            blocked2 = True
                            break
                    if blocked2:
                        continue
                    phi = angle_deg(ghost - cue_pos)
                    d = np.linalg.norm(ghost - cue_pos)
                    V0 = float(np.clip(1.2 + 0.8 * d, 0.5, 6.0))
                    # 不考虑杆头高度和偏移
                    theta = 0.0
                    a = 0.0
                    b = 0.0
                    candidates.append({'V0': V0, 'phi': phi, 'theta': theta, 'a': a, 'b': b})
            
            # 如果没有候选动作，尝试直接击打最近的目标球
            if len(candidates) == 0:
                best_target = None
                best_d = 1e9
                for bid in my_targets:
                    if balls[bid].state.s == 4:
                        continue
                    tgt = vec2(balls[bid].state.rvw[0])
                    d = np.linalg.norm(tgt - cue_pos)
                    if d < best_d:
                        best_d = d
                        best_target = bid
                if best_target is None:
                    return self._random_action()
                tgt = vec2(balls[best_target].state.rvw[0])
                phi = angle_deg(tgt - cue_pos)
                V0 = float(np.clip(1.2 + 0.6 * best_d, 0.5, 6.0))
                theta = 0.0
                a = 0.0
                b = 0.0
                return {'V0': V0, 'phi': phi, 'theta': theta, 'a': a, 'b': b}
            
            # 扰动搜索
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            best_action = None
            best_score = -1e9
            cand_n = len(candidates)
            if cand_n <= 2:
                dphi_grid = [-4.0, -2.0, 0.0, 2.0, 4.0]
                dv_grid = [-0.5, -0.2, 0.0, 0.2, 0.5]
                da_grid = [-0.06, -0.03, 0.0, 0.03, 0.06]
                db_grid = [-0.06, -0.03, 0.0, 0.03, 0.06]
            elif cand_n >= 6:
                dphi_grid = [-1.5, 0.0, 1.5]
                dv_grid = [-0.2, 0.0, 0.2]
                da_grid = [0.0]
                db_grid = [0.0]
            else:
                dphi_grid = [-2.0, 0.0, 2.0]
                dv_grid = [-0.3, 0.0, 0.3]
                da_grid = [-0.04, 0.0, 0.04]
                db_grid = [-0.04, 0.0, 0.04]
            for base in candidates:
                act = {'V0': base['V0'],
                       'phi': base['phi'],
                       'theta': base['theta'],
                       'a': base['a'],
                       'b': base['b']}
                for dphi in dphi_grid:
                    for dv in dv_grid:
                        for da in da_grid:
                            for db in db_grid:
                                act = {
                                    'V0': float(np.clip(base['V0'] + dv, 0.5, 8.0)),
                                    'phi': (base['phi'] + dphi) % 360.0,
                                    'theta': base['theta'],
                                    'a': float(np.clip(base['a'] + da, -0.5, 0.5)),
                                    'b': float(np.clip(base['b'] + db, -0.5, 0.5)),
                                }
                        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                        sim_table = copy.deepcopy(table)
                        cue = pt.Cue(cue_ball_id="cue")
                        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                        try:
                            shot.cue.set_state(V0=act['V0'], phi=act['phi'], theta=act['theta'], a=act['a'], b=act['b'])
                            pt.simulate(shot, inplace=True)
                            score = analyze_shot_for_reward(shot=shot, last_state=last_state_snapshot, player_targets=my_targets)
                        except Exception:
                            # 模拟失败，给予极大惩罚
                            score = -500.0
                        if score > best_score:
                            best_score = score
                            best_action = act
            # 选择得分最高的动作
            if best_action is None or best_score < -100.0:
                return self._random_action()
            return best_action
        except Exception as e:
            print(f"[NewAgent] 发生异常，返回随机动作。异常信息：{e}")
            return self._random_action()
        

# ============ MCTS 框架 ============

class MCTSNode:
    """蒙特卡洛树搜索节点"""
    def __init__(self, state, parent=None, action=None):
        """
        参数：
            state: 当前状态 (balls, my_targets, table)
            parent: 父节点
            action: 导致该状态的动作 dict或None
        """
        self.state = state  # 当前状态的快照
        self.action = action  # 导致该状态的动作
        self.parent = parent  # 父节点
        self.children = {}  # 子节点字典 {action_key: MCTSNode}
        self.visits = 0  # 访问次数 N(s,a)
        self.value = 0.0  # 累积奖励 Q(s,a)
        self.untried_actions = None  # 未尝试的动作集合
    
    def ucb_score(self, c=1.414):
        """
        计算UCB1分数（Upper Confidence Bound）
        
        UCB1 = Q(s,a) / N(s,a) + c * sqrt(ln(N(parent)) / N(s,a))
        """
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        if self.parent is None or self.parent.visits == 0:
            exploration = 0
        else:
            exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
    
    def is_fully_expanded(self):
        """检查是否已完全扩展（所有动作都被尝试过）"""
        return len(self.untried_actions) == 0 if self.untried_actions is not None else False
    
    def best_child(self, c=1.414):
        """选择最优子节点（UCB1最高）"""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda child: child.ucb_score(c))
    
    def select_untried_action(self):
        """随机选择一个未尝试的动作"""
        if self.untried_actions is None or len(self.untried_actions) == 0:
            return None
        action = random.choice(self.untried_actions)
        self.untried_actions.remove(action)
        return action

# ============ 防守策略模块 ============

class DefenseMode(Enum):
    """台球防守模式枚举"""
    AGGRESSIVE = 0      # 进攻模式：追求最大进球得分
    BALANCED = 1        # 平衡模式：平衡攻防
    DEFENSIVE = 2       # 防守模式：重点阻止对手得分


def calculate_defensive_score(balls: dict, my_targets: list, 
                             opponent_targets: list, action: dict, 
                             shot: pt.System, table,
                             distance_weight=0.15, blocking_weight=0.10, 
                             position_weight=0.10, scatter_weight=0.25,
                             path_blocking_weight=0.25, critical_weight=0.10,
                             safety_weight=0.05):
    """
    计算一个动作的防守价值分数（不考虑进球得分）
    
    参数：
        balls: 击球前的球状态
        my_targets: 我的目标球
        opponent_targets: 对手的目标球
        action: 本次动作 {'V0', 'phi', 'theta', 'a', 'b'}
        shot: 模拟后的System对象（已完成物理模拟）
        table: 台面信息
        distance_weight: 距离阻断权重 (default: 0.15)
        blocking_weight: 球数阻断权重 (default: 0.10)
        position_weight: 位置防守权重 (default: 0.10)
        scatter_weight: 球群分散度权重 (default: 0.25) ⭐新增
        path_blocking_weight: 路线阻挡权重 (default: 0.25) ⭐新增
        critical_weight: 关键球防守权重 (default: 0.10)
        safety_weight: 白球安全距离权重 (default: 0.05)
    
    返回：
        float: 防守得分 (0-100)
            计算七个防守指标：
            1. 距离阻断：对手目标球与袋口的平均距离
            2. 球数阻断：被遮挡的对手目标球数
            3. 位置防守：白球的防守深度
            4. 球群分散度：对手目标球分布紧凑程度 ⭐新增
            5. 路线阻挡：白球对多个对手进球路线的阻挡 ⭐新增
            6. 关键球防守：白球是否防守容易进的球
            7. 白球安全性：白球是否在安全距离
    """
    try:
        from utils import vec2, norm, angle_deg
        
        defensive_score = 0
        
        # ========== 指标1：距离阻断 ==========
        # 计算对手目标球到最近袋口的距离
        opponent_remaining = [bid for bid in opponent_targets 
                                if shot.balls[bid].state.s != 4]  # 未进袋
        
        if opponent_remaining:
            pockets = list(table.pockets.values())
            min_distances = []
            
            for bid in opponent_remaining:
                ball_pos = vec2(shot.balls[bid].state.rvw[0])
                min_dist_to_pocket = min(
                    np.linalg.norm(vec2(pk.center) - ball_pos)
                    for pk in pockets
                )
                min_distances.append(min_dist_to_pocket)
            
            # 距离越远越好（平均距离标准化到0-100）
            avg_distance = np.mean(min_distances) if min_distances else 0
            distance_score = min(100, avg_distance * 15)  # 约6.7m = 100分
            defensive_score += distance_weight * distance_score
        
        # ========== 指标2：球数阻断 ==========
        # 计算有多少对手目标球被我方球遮挡
        cue_pos = vec2(shot.balls['cue'].state.rvw[0])
        R = 0.028575  # 球的半径
        
        blocked_count = 0
        for bid in opponent_remaining:
            target_pos = vec2(shot.balls[bid].state.rvw[0])
            pockets = list(table.pockets.values())
            
            # 检查此球是否被当前白球位置"遮挡"（相对于最近的袋口）
            nearest_pocket = min(pockets, 
                                key=lambda pk: np.linalg.norm(vec2(pk.center) - target_pos))
            pocket_center = vec2(nearest_pocket.center)
            
            # 判断白球是否在"目标球→袋口"的进球路线上
            # 使用点到线段的距离：如果距离 < 球的半径，则认为有阻挡
            path_dir = norm(pocket_center - target_pos)  # 进球方向单位向量
            cue_to_target = cue_pos - target_pos  # 白球相对于目标球的向量
            
            # 计算白球在进球方向上的投影长度
            projection_length = np.dot(cue_to_target, path_dir)
            
            # 白球在目标球和袋口之间，且距离进球路线足够近
            if projection_length > 0 and projection_length < np.linalg.norm(pocket_center - target_pos):
                # 计算白球到进球路线的垂直距离
                projection_pos = target_pos + projection_length * path_dir
                perp_dist = np.linalg.norm(cue_pos - projection_pos)
                
                if perp_dist < 2.1 * R:  # 2.1倍球半径的容差
                    blocked_count += 1
        
        blocking_score = (blocked_count / max(1, len(opponent_remaining))) * 100
        defensive_score += blocking_weight * blocking_score
        
        # ========== 指标3：位置防守 ==========
        # 白球停留在对手球堆附近更好（防守深度）
        # 计算白球到对手目标球群的距离
        if opponent_remaining:
            opponent_positions = [
                vec2(shot.balls[bid].state.rvw[0]) 
                for bid in opponent_remaining
            ]
            
            # 对手球堆中心
            opponent_center = np.mean([p for p in opponent_positions], axis=0)
            
            # 白球到对手球堆的距离（越近越好用于防守）
            cue_pos = vec2(shot.balls['cue'].state.rvw[0])
            dist_to_opponent = np.linalg.norm(cue_pos - opponent_center)
            
            # 距离标准化：约1.5m以内为最优防守位置
            position_score = min(100, 100 * (1.5 / max(0.1, dist_to_opponent)))
            defensive_score += position_weight * position_score
        
        # ========== 指标4：球群分散度 ==========
        # 对手目标球分布越紧凑，防守越容易
        if opponent_remaining and len(opponent_remaining) >= 2:
            opponent_positions = [
                vec2(shot.balls[bid].state.rvw[0]) 
                for bid in opponent_remaining
            ]
            
            # 计算球群的方差（紧凑度指标）
            opponent_positions_array = np.array([list(p) for p in opponent_positions])
            center = np.mean(opponent_positions_array, axis=0)
            
            # 计算每个球到中心的距离
            distances_to_center = [
                np.linalg.norm(pos - center) 
                for pos in opponent_positions_array
            ]
            
            # 方差越小 = 球越紧凑 = 防守越容易
            scatter_variance = np.var(distances_to_center) if distances_to_center else 0
            
            # 标准化：方差0~0.3m² → 100~0分
            # 方差越大（球越分散），防守得分越低
            scatter_score = max(0, 100 - scatter_variance * 333)  # 333 = 100/0.3
            defensive_score += scatter_weight * scatter_score
        
        # ========== 指标5：路线阻挡 ==========
        # 白球对对手进球路线的阻挡程度
        if opponent_remaining:
            cue_pos = vec2(shot.balls['cue'].state.rvw[0])
            pockets = list(table.pockets.values())
            R = 0.028575
            
            # 计算白球对每个对手球的路线阻挡程度
            total_path_blocking = 0.0
            
            for bid in opponent_remaining:
                target_pos = vec2(shot.balls[bid].state.rvw[0])
                
                # 找到此球最近的袋口
                nearest_pocket = min(pockets, 
                                    key=lambda pk: np.linalg.norm(vec2(pk.center) - target_pos))
                pocket_center = vec2(nearest_pocket.center)
                
                # 计算白球在进球路线上的阻挡程度（0~1）
                path_dir = norm(pocket_center - target_pos)
                cue_to_target = cue_pos - target_pos
                
                # 投影长度（白球沿进球方向的位置）
                projection_length = np.dot(cue_to_target, path_dir)
                path_length = np.linalg.norm(pocket_center - target_pos)
                
                # 如果白球在进球路线上（0 < 投影 < 路径长度）
                if 0 < projection_length < path_length:
                    # 计算垂直距离（越近阻挡越强）
                    projection_pos = target_pos + projection_length * path_dir
                    perp_dist = np.linalg.norm(cue_pos - projection_pos)
                    
                    # 垂直距离 < 10R时，认为有阻挡
                    if perp_dist < 10 * R:
                        # 阻挡强度 = 1 - (距离 / 10R)，范围0~1
                        blocking_strength = 1.0 - (perp_dist / (10 * R))
                        total_path_blocking += blocking_strength
            
            # 路线阻挡得分：标准化到0~100
            # 如果能同时阻挡所有对手球的路线 = 100分
            max_possible_blocking = len(opponent_remaining)
            path_blocking_score = (total_path_blocking / max(1, max_possible_blocking)) * 100
            defensive_score += path_blocking_weight * path_blocking_score
        
        return defensive_score
        
    except Exception as e:
        # 防守计算失败时返回0分
        print(f"[MCTSAgent] 计算防守分数时发生错误：{e}")
        return 0


class MCTSAgent(Agent):
    """基于蒙特卡洛树搜索的 Agent"""
    
    def __init__(self, num_iterations=50, exploration_c=1.414, 
                 use_heuristic=True, verbose=False, use_2step=True, 
                 opponent_weight=0.7, use_defense=False, 
                 defense_threshold=0.4, defense_weight=0.3):
        """
        参数：
            num_iterations: MCTS迭代次数
            exploration_c: UCB1探索系数
            use_heuristic: 是否使用启发式生成动作候选
            verbose: 是否打印调试信息
            use_2step: 是否使用2步前向搜索（考虑对手回应）
            opponent_weight: 对手得分的权重系数（0.5-1.0）
            use_defense: 是否启用防守策略 (default: False)
            defense_threshold: 触发防守模式的阈值 (default: 0.4, 0-1)
            defense_weight: 防守得分的权重 (default: 0.3, 0-1)
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.exploration_c = exploration_c
        self.use_heuristic = use_heuristic
        self.verbose = verbose
        self.use_2step = use_2step
        self.opponent_weight = opponent_weight
        # 防守策略参数
        self.use_defense = use_defense
        self.defense_threshold = defense_threshold
        self.defense_weight = defense_weight
        self.new_agent = NewAgent()  # 用于启发式候选和快速模拟
    
    def _generate_action_candidates(self, balls, my_targets, table, num_samples=10):
        """
        生成动作候选集合（优化版）
        
        【方案A】智能优先级排序 - 评分并排序候选
        【方案B】动态候选数量 - 根据剩余球数调整生成策略
        
        返回：list of dicts，每个dict是一个动作
        """
        try:
            from utils import vec2, norm, angle_deg
            
            candidates = []
            
            # 方式1: 如果启用启发式，先用NewAgent生成候选
            if self.use_heuristic:
                R = 0.028575
                cue_pos = vec2(balls["cue"].state.rvw[0])
                pockets = list(table.pockets.values())
                
                # 【方案B】根据剩余球数动态调整策略
                my_remaining = len([b for b in my_targets if b != '8' and balls[b].state.s != 4])
                
                if my_remaining <= 3:
                    # 清台阶段：只看最容易进的球，减少计算
                    max_targets = 2
                    max_pockets_per_target = 2
                else:
                    # 早期阶段：多样化探索
                    max_targets = min(3, len(my_targets))
                    max_pockets_per_target = 3
                
                # 步骤1：计算所有目标球的"容易进"程度
                target_scores = []
                for bid in my_targets:
                    if balls[bid].state.s == 4:
                        continue
                    
                    tgt = vec2(balls[bid].state.rvw[0])
                    
                    # 计算此球到最近袋口的距离
                    min_dist_to_pocket = min(
                        np.linalg.norm(vec2(pk.center) - tgt)
                        for pk in pockets
                    )
                    
                    # 容易进程度：距离越近越容易
                    easiness = max(0, 1.0 - min_dist_to_pocket / 1.5)  # 1.5m为参考距离
                    target_scores.append((bid, tgt, easiness))
                
                # 排序：容易进的球优先
                target_scores.sort(key=lambda x: x[2], reverse=True)
                
                # 步骤2：为前max_targets个目标球生成候选
                for bid, tgt, easiness in target_scores[:max_targets]:
                    # 找到最近的max_pockets_per_target个袋口
                    pocket_distances = [
                        (pk, np.linalg.norm(vec2(pk.center) - tgt))
                        for pk in pockets
                    ]
                    pocket_distances.sort(key=lambda x: x[1])
                    
                    for pk, dist_to_pocket in pocket_distances[:max_pockets_per_target]:
                        pkc = vec2(pk.center)
                        dir_tp = norm(pkc - tgt)
                        ghost = tgt - 2.0 * R * dir_tp
                        
                        if np.linalg.norm(cue_pos - ghost) < 1e-6:
                            continue
                        
                        phi = angle_deg(ghost - cue_pos)
                        d = np.linalg.norm(ghost - cue_pos)
                        V0 = float(np.clip(1.2 + 0.8 * d, 0.5, 6.0))
                        
                        # 【方案A】为候选评分
                        action = {
                            'V0': V0, 'phi': phi, 'theta': 0.0, 
                            'a': 0.0, 'b': 0.0
                        }
                        
                        # 评分公式：容易进程度 + V0合理性
                        optimal_d = 0.5  # 理想击球距离
                        v0_reasonableness = max(0, 1.0 - abs(d - optimal_d) / optimal_d)
                        score = easiness * 60 + v0_reasonableness * 40
                        
                        candidates.append({
                            'action': action,
                            'score': score,
                            'phi': phi
                        })
            
            # 步骤3：【方案A】排序并去重
            if candidates:
                # 按评分排序
                candidates.sort(key=lambda x: x['score'], reverse=True)
                
                # 过滤方向相似的候选（避免重复）
                filtered = []
                for cand in candidates:
                    # 检查是否与已有候选的方向过于相似
                    is_similar = False
                    for existing in filtered:
                        phi_diff = abs(cand['phi'] - existing['phi'])
                        # 处理 360度 回绕
                        if phi_diff > 180:
                            phi_diff = 360 - phi_diff
                        
                        if phi_diff < 15:  # 方向差<15度认为重复
                            is_similar = True
                            break
                    
                    if not is_similar:
                        filtered.append(cand)
                
                candidates = filtered
            
            # 步骤4：添加随机采样保证多样性
            num_heuristic = len(candidates)
            num_random = max(1, num_samples - num_heuristic)
            
            for _ in range(num_random):
                candidates.append({
                    'action': self._random_action(),
                    'score': 0,  # 随机动作评分为0
                    'phi': 0
                })
            
            # 步骤5：返回前num_samples个
            return [c['action'] for c in candidates[:num_samples]]
            
        except Exception as e:
            # 异常时回退到随机采样
            if self.verbose:
                print(f"[MCTS] 候选生成异常: {e}，使用随机动作")
            return [self._random_action() for _ in range(num_samples)]
    
    
    def _action_to_key(self, action):
        """将动作转换为字典键（便于在字典中查找）"""
        return tuple(action[k] for k in ['V0', 'phi', 'theta', 'a', 'b'])
    
    def _infer_opponent_targets(self, balls, my_targets):
        """
        推断对手的目标球
        
        参数：
            balls: 球状态
            my_targets: 我的目标球
        
        返回：list, 对手的目标球ID列表
        """
        all_solid = ['1', '2', '3', '4', '5', '6', '7']
        all_stripe = ['9', '10', '11', '12', '13', '14', '15']
        
        # 判断我的目标是什么
        if my_targets == ['8']:
            # 我已清台，对手目标未知，返回所有可能
            remaining_solid = [bid for bid in all_solid if balls[bid].state.s != 4]
            remaining_stripe = [bid for bid in all_stripe if balls[bid].state.s != 4]
            return remaining_solid if len(remaining_solid) > 0 else remaining_stripe
        
        # 检查我是实心还是条纹
        my_is_solid = any(bid in all_solid for bid in my_targets)
        
        if my_is_solid:
            # 我是实心，对手是条纹
            opponent_targets = [bid for bid in all_stripe if balls[bid].state.s != 4]
        else:
            # 我是条纹，对手是实心
            opponent_targets = [bid for bid in all_solid if balls[bid].state.s != 4]
        
        # 如果对手已清台，目标是黑8
        if len(opponent_targets) == 0:
            opponent_targets = ['8']
        
        return opponent_targets
    
    def _select_defense_mode(self, balls, my_targets, opponent_targets):
        """
        【层2：防守模式选择】根据当前游戏状态选择防守模式
        
        参数：
            balls: 当前球状态
            my_targets: 我的目标球
            opponent_targets: 对手的目标球
        
        返回：
            DefenseMode: 选择的防守模式
        
        触发逻辑：
            - 进攻模式：我的剩余目标球 ≤ 2 个（自动清台优先级最高）
            - 防守模式：对手剩余目标球 ≤ 2 个（防止对手清台）
            - 平衡模式：其他情况
        """
        # 统计剩余目标球数
        my_remaining = sum(1 for bid in my_targets 
                          if balls[bid].state.s != 4)
        opp_remaining = sum(1 for bid in opponent_targets 
                           if balls[bid].state.s != 4)
        
        # 优先级1：我即将清台 -> 进攻
        if my_remaining <= 2:
            if self.verbose:
                print(f"[MCTS] 选择防守模式：进攻 (我剩余目标球: {my_remaining})")
            return DefenseMode.AGGRESSIVE
        
        # 优先级2：对手即将清台 -> 防守
        if opp_remaining <= 2:
            if self.verbose:
                print(f"[MCTS] 选择防守模式：防守 (对手剩余目标球: {opp_remaining})")
            return DefenseMode.DEFENSIVE
        
        # 优先级3：其他情况 -> 平衡
        if self.verbose:
            print(f"[MCTS] 选择防守模式：平衡 (我剩余目标球: {my_remaining}, 对手剩余目标球: {opp_remaining})")
        return DefenseMode.BALANCED
    
    def _evaluation_position(self, shot, my_targets, opponent_targets, table):
        """
        评估白球停留位置的质量
        
        参数：
            shot: 模拟后的 System 对象
            my_targets: 我的目标球
            opponent_targets: 对手的目标球
            table: 击球后的台面信息
        
        返回：
            float: 位置评分 (-10 到 +15)
        """
        try:
            from utils import vec2
            
            score = 0.0
            
            # 获取白球位置
            cue_pos = vec2(shot.balls['cue'].state.rvw[0])
            
            # 1. 白球距离对手目标球的距离评分 (+0 到 +10)
            if opponent_targets:
                opponent_positions = []
                for bid in opponent_targets:
                    if shot.balls[bid].state.s != 4:  # 未进袋
                        opponent_positions.append(vec2(shot.balls[bid].state.rvw[0]))
                
                if opponent_positions:
                    # 计算白球到对手球群的平均距离
                    avg_dist = np.mean([np.linalg.norm(cue_pos - opp_pos) 
                                       for opp_pos in opponent_positions])
                    # 距离越远越好（>1.5m 为最佳）
                    dist_score = min(10.0, avg_dist * 6.0)
                    score += dist_score
            
            # 2. 白球安全性评分 (-10 到 +5)
            # 检查白球是否贴库（不好的位置）
            R = 0.028575  # 球半径
            table_width = table.w
            table_length = table.l
            
            # 动态阈值（基于球半径和球桌尺寸）
            very_close_threshold = 5 * R      # ≈ 0.143m，非常贴库
            close_threshold = 10 * R          # ≈ 0.286m，靠近边界
            safe_threshold = max(15 * R, min(table_width, table_length) * 0.15)  # 至少15R或球桌短边的15%
            
            # 白球到边界的最小距离
            margin_x = min(cue_pos[0], table_width - cue_pos[0])
            margin_y = min(cue_pos[1], table_length - cue_pos[1])
            min_margin = min(margin_x, margin_y)
            
            # 如果白球太靠近边界，扣分
            if min_margin < very_close_threshold:
                score -= 10.0
            elif min_margin < close_threshold:
                score -= 5.0
            elif min_margin > safe_threshold:
                # 白球在台面中央区域，加分
                score += 5.0
            
            return score
            
        except Exception as e:
            # 评估失败，返回0分
            if self.verbose:
                print(f"[MCTSAgent] 评估位置分数时发生错误：{e}")
                import traceback
                traceback.print_exc()
            return 0.0
    
    def _calculate_black_ball_risk(self, shot: pt.System, my_targets: list, table) -> float:
        """
        计算黑8球风险分数（不包括首球犯规，首球犯规在_analyze_shot中已处理）
        
        参数：
            shot: 模拟后的 System 对象
            my_targets: 我的目标球列表
            table: 击球后的台面信息
        
        返回：
            float: 风险分数 (0-10)，越高表示风险越大
        """
        try:
            from utils import vec2
            
            # 只在清台前评估黑8风险
            if '8' in my_targets:
                return 0.0
            
            risk_score = 0.0
            
            # 1. 检查黑8击球后的位置风险
            if '8' in shot.balls and shot.balls['8'].state.s != 4:  # 黑8未进袋
                black_8_pos = vec2(shot.balls['8'].state.rvw[0])
                pockets = list(table.pockets.values())
                
                # 找到最近的袋口
                min_dist_to_pocket = min(
                    np.linalg.norm(vec2(pk.center) - black_8_pos)
                    for pk in pockets
                )
                
                R = 0.028575  # 球半径
                # 黑8非常接近袋口（意外进袋风险）
                if min_dist_to_pocket < 5 * R:  # < 0.143m
                    risk_score += 8.0  # 高风险：黑8可能被后续球碰进
                elif min_dist_to_pocket < 8 * R:  # < 0.229m
                    risk_score += 4.0   # 中等风险
                elif min_dist_to_pocket < 12 * R:  # < 0.343m
                    risk_score += 2.0   # 低风险
            
            return risk_score
            
        except Exception as e:
            # 计算失败时返回0分
            if self.verbose:
                print(f"[MCTSAgent] 计算黑8风险时发生错误：{e}")
            return 0.0
        
    def _analyze_shot(self, shot: pt.System, last_state: dict, player_targets: list, table) -> float:
        """
        分析一次击球的结果，计算奖励分数
        基于 analyze_shot_for_reward 函数
        
        参数:
            shot: 模拟后的 System 对象
            last_state: 击球前的球状态快照
            player_targets: 我的目标球列表
            table: 击球后的台面信息
        """
        # 1. 基本分析
        opponent_targets = self._infer_opponent_targets(shot.balls, player_targets)
        
        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
        
        # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
        own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
        enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
        
        my_remaining = len([b for b in player_targets if b != '8' and shot.balls[b].state.s != 4])
        opp_remaining = len([b for b in opponent_targets if b != '8' and shot.balls[b].state.s != 4])

        
        cue_pocketed = "cue" in new_pocketed
        eight_pocketed = "8" in new_pocketed

        # 2. 分析首球碰撞（定义合法的球ID集合）
        first_contact_ball_id = None
        foul_first_hit = False
        valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
        
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
                other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break
        
        # 首球犯规判定：完全对齐 player_targets
        if first_contact_ball_id is None:
            # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
            if len(last_state) > 2 or player_targets != ['8']:
                foul_first_hit = True
        else:
            # 首次击打的球必须是 player_targets 中的球
            if first_contact_ball_id not in player_targets:
                foul_first_hit = True
        
        # 3. 分析碰库
        cue_hit_cushion = False
        target_hit_cushion = False
        foul_no_rail = False
        
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if 'cushion' in et:
                if 'cue' in ids:
                    cue_hit_cushion = True
                if first_contact_ball_id is not None and first_contact_ball_id in ids:
                    target_hit_cushion = True

        if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
            foul_no_rail = True
            
        # 4. 计算奖励分数
        score = 0
        if my_remaining <= 2:
            pocket_reward = 100
        elif my_remaining >= 6:
            pocket_reward = 60
        else:
            pocket_reward = 75 
        
        # 白球进袋处理
        if cue_pocketed and eight_pocketed:
            score -= 150  # 白球+黑8同时进袋，严重犯规
        elif cue_pocketed:
            score -= 100  # 白球进袋
        elif eight_pocketed:
            # 黑8进袋：只有清台后（player_targets == ['8']）才合法
            if player_targets == ['8']:
                score += 100  # 合法打进黑8
            else:
                score -= 150  # 清台前误打黑8，判负
                
        # 首球犯规和碰库犯规
        if foul_first_hit:
            if first_contact_ball_id == '8':
                score -= 50  # 首球犯规且打到黑8，严重犯规
            else:
                score -= 25
        if foul_no_rail:
            score -= 25
            
        # 进球得分（own_pocketed 已根据 player_targets 正确分类）
        score += len(own_pocketed) * pocket_reward
        score -= len(enemy_pocketed) * 20
        
        # 合法无进球
        if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
            # 根据停球位置给予少量奖励
            position_score = self._evaluation_position(shot, player_targets, opponent_targets, table)
            score = 5 + position_score
            
        # 黑8风险评估
        if '8' not in player_targets:
            black_ball_risk = self._calculate_black_ball_risk(shot, player_targets, table)
            score -= black_ball_risk  
            
        return score
    
    def _simulate_action(self, balls, table, action, my_targets, opponent_targets=None):
        """
        快速模拟一个击球动作，返回奖励
        
        【Layer 3】支持防守评分
        
        参数：
            balls: 球状态
            table: 球桌
            action: 击球动作
            my_targets: 我的目标球
            opponent_targets: 对手的目标球（用于防守评分计算）
        
        返回：float, 该动作的综合奖励得分
        """
        try:
            last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            
            shot.cue.set_state(
                V0=action['V0'], 
                phi=action['phi'], 
                theta=action['theta'], 
                a=action['a'], 
                b=action['b']
            )
            
            # 使用超时保护
            if not simulate_with_timeout(shot, timeout=3):
                return 0  # 超时返回0分
            
            
            # 计算基础奖励（进球得分）
            reward = self._analyze_shot(shot, last_state, my_targets, sim_table)
            
            # 【Layer 3】如果启用防守策略，加入防守评分
            if self.use_defense and opponent_targets is not None:
                try:
                    defense_score = calculate_defensive_score(
                        balls=balls,
                        my_targets=my_targets,
                        opponent_targets=opponent_targets,
                        action=action,
                        shot=shot,
                        table=table
                    )
                    # 防守得分加权（权重范围0-1）
                    reward = reward + self.defense_weight * defense_score
                    
                    if self.verbose:
                        print(f"[MCTS] 基础奖励: {reward-self.defense_weight*defense_score:.1f}, 防守得分: {defense_score:.1f}, 综合: {reward:.1f}")
                except Exception as e:
                    if self.verbose:
                        print(f"[MCTS] 防守评分失败: {e}")
                    # 防守评分失败，仅使用基础奖励
            
            return reward
            
        except Exception as e:
            if self.verbose:
                print(f"[MCTS] 模拟失败: {e}")
            return -500
    
    def _simulate_action_2step(self, balls, table, my_action, my_targets, opponent_targets):
        """
        2步前向搜索：模拟我的击球 + 对手的最佳回应
        
        参数：
            balls: 球状态
            table: 球桌
            my_action: 我的击球动作
            my_targets: 我的目标球
            opponent_targets: 对手的目标球
        
        返回：
            float, 综合评分 = my_reward - opponent_weight * opponent_best_reward
        """
        try:
            # 步骤1：模拟我的击球
            last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            
            shot.cue.set_state(
                V0=my_action['V0'], 
                phi=my_action['phi'], 
                theta=my_action['theta'], 
                a=my_action['a'], 
                b=my_action['b']
            )
            
            # 使用超时保护
            if not simulate_with_timeout(shot, timeout=3):
                return 0
            
            # 步骤1：计算我的基础得分（不包含防守得分）
            my_base_reward = self._analyze_shot(shot, last_state, my_targets, sim_table)
            
            # 获取我击球后的球状态
            balls_after_my_shot = shot.balls
            
            # 步骤2：对手的最佳回应
            # 检查对手是否还有球
            opponent_remaining = [bid for bid in opponent_targets 
                                if balls_after_my_shot[bid].state.s != 4]
            
            if len(opponent_remaining) == 0:
                # 对手已清台，只考虑黑8
                opponent_targets = ["8"]
            else:
                opponent_targets = opponent_remaining
            
            # 为对手生成少量候选动作（3-5个）
            opponent_actions = self._generate_action_candidates(
                balls_after_my_shot, opponent_targets, table, 
                num_samples=5  # 减少对手的采样数，加快速度
            )
            
            if not opponent_actions:
                # 对手无法行动，返回我的综合得分
                if self.use_defense:
                    defense_score = calculate_defensive_score(
                        balls=balls,
                        my_targets=my_targets,
                        opponent_targets=opponent_targets,
                        action=my_action, 
                        shot=shot,
                        table=table
                    )
                    my_reward = my_base_reward + self.defense_weight * defense_score
                else:
                    my_reward = my_base_reward
                return my_reward
            
            # 对手选择最好的动作
            best_opponent_reward = -1e9
            for opp_action in opponent_actions:
                # 对手的simulate_action已包含防守得分
                opp_reward = self._simulate_action(
                    balls_after_my_shot, table, opp_action, opponent_targets
                )
                best_opponent_reward = max(best_opponent_reward, opp_reward)
            
            # 步骤3：【Layer 3】计算我方综合得分 = 基础得分 + 防守得分
            if self.use_defense:
                # 计算防守得分
                defense_score = calculate_defensive_score(
                    balls=balls,
                    my_targets=my_targets,
                    opponent_targets=opponent_targets,
                    action=my_action,
                    shot=shot,
                    table=table
                )
                my_reward = my_base_reward + self.defense_weight * defense_score
            else:
                my_reward = my_base_reward
            
            # 步骤4：【Layer 3】根据防守模式调整综合评分公式
            if self.use_defense:
                # 动态模式选择
                defense_mode = self._select_defense_mode(
                    balls_after_my_shot, my_targets, opponent_targets
                )
                
                if defense_mode == DefenseMode.AGGRESSIVE:
                    # 进攻模式：我方即将清台，不太关心对手威胁
                    # 公式: (基础得分 + 防守) - 0.3 * 对手得分
                    # （权重低，对手威胁影响最小）
                    final_score = my_reward - 0.3 * best_opponent_reward
                    mode_name = "AGGRESSIVE"
                    
                elif defense_mode == DefenseMode.DEFENSIVE:
                    # 防守模式：对手即将清台，非常关心对手威胁
                    # 公式: (基础得分 + 防守) - 0.95 * 对手得分
                    # （权重高，对手威胁影响最大，防守优先）
                    final_score = my_reward - 0.85 * best_opponent_reward
                    mode_name = "DEFENSIVE"
                    
                else:  # BALANCED
                    # 平衡模式：正常情况，中等权重
                    # 公式: (基础得分 + 防守) - opponent_weight * 对手得分
                    final_score = my_reward - self.opponent_weight * best_opponent_reward
                    mode_name = "BALANCED"
                
                if self.verbose:
                    print(f"[2-Step] 模式: {mode_name}, 基础: {my_base_reward:.1f}, 防守: {defense_score:.1f}, 综合我: {my_reward:.1f}, 对手最佳: {best_opponent_reward:.1f}, 最终: {final_score:.1f}")
            else:
                # 不使用防守策略，使用原始公式
                final_score = my_reward - self.opponent_weight * best_opponent_reward
                
                if self.verbose:
                    print(f"[2-Step] 基础得分: {my_reward:.1f}, 对手最佳: {best_opponent_reward:.1f}, 综合: {final_score:.1f}")
            
            return final_score
            
        except Exception as e:
            if self.verbose:
                print(f"[MCTS 2-Step] 模拟失败: {e}")
            return -500
    
    def _selection(self, node):
        """
        选择阶段：从根节点选择到叶节点
        
        返回完全展开的节点或叶节点
        """
        while not node.is_fully_expanded():
            return node
        
        # 所有子节点都已扩展，选择最优子节点继续
        child = node.best_child(c=self.exploration_c)
        if child is None:
            return node
        return self._selection(child)
    
    def _expansion(self, node, balls, my_targets, table):
        """
        扩展阶段：从选中的节点添加新子节点
        
        返回：新添加的子节点（或现有节点）
        """
        # 如果节点的未尝试动作集合为None，初始化它
        if node.untried_actions is None:
            node.untried_actions = self._generate_action_candidates(
                balls, my_targets, table, 
                num_samples=10
            )
        
        # 选择一个未尝试的动作
        if len(node.untried_actions) == 0:
            return node
        
        action = node.select_untried_action()
        if action is None:
            return node
        
        # 创建新子节点
        # 注意：这里我们不真正进行深层模拟，只记录动作
        action_key = self._action_to_key(action)
        new_child = MCTSNode(state=node.state, parent=node, action=action)
        node.children[action_key] = new_child
        
        return new_child
    
    def _simulation(self, node, balls, my_targets, table, opponent_targets=None):
        """
        模拟阶段：快速评估该节点的价值
        
        【Layer 3】支持防守评分的动态评估
        
        根据use_2step和use_defense参数决定模拟方式：
        - use_2step=True: 2步前向搜索（考虑对手回应）
        - use_defense=True: 加入防守评分
        
        返回：float, 该节点的奖励值
        """
        if node.action is None:
            # 根节点，不执行任何动作
            return 0
        
        # 根据配置选择模拟方式
        if self.use_2step and opponent_targets is not None:
            # 使用2步前向搜索（考虑对手回应）
            reward = self._simulate_action_2step(
                balls, table, node.action, my_targets, opponent_targets
            )
        else:
            # 使用1步模拟
            # 【Layer 3】如果启用防守，传递opponent_targets用于防守评分
            if self.use_defense and opponent_targets is not None:
                reward = self._simulate_action(
                    balls, table, node.action, my_targets, opponent_targets
                )
            else:
                reward = self._simulate_action(
                    balls, table, node.action, my_targets, None
                )
        
        return reward
    
    def _backpropagation(self, node, reward):
        """
        反向传播阶段：更新节点的访问次数和累积奖励
        """
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def search(self, balls, my_targets, table, opponent_targets=None):
        """
        执行MCTS搜索，返回最优动作
        
        参数：
            balls: 球状态字典
            my_targets: 目标球列表
            table: 球桌对象
            opponent_targets: 对手的目标球列表（用于2步搜索）
        
        返回：dict, 最优击球动作
        """
        if self.verbose:
            mode_str = "2步搜索" if (self.use_2step and opponent_targets) else "1步搜索"
            print(f"[MCTS] 开始搜索 (iterations={self.num_iterations}, mode={mode_str})")
        
        # 如果启用2步搜索但没有提供对手目标球，自动推断
        if self.use_2step and opponent_targets is None:
            opponent_targets = self._infer_opponent_targets(balls, my_targets)
            if self.verbose:
                print(f"[MCTS] 推断对手目标球: {opponent_targets}")
        
        # 初始化根节点
        root = MCTSNode(state=(balls, my_targets, table), parent=None)
        root.untried_actions = self._generate_action_candidates(
            balls, my_targets, table, 
            num_samples=10
        )
        
        # 执行迭代
        for iteration in range(self.num_iterations):
            # 1. 选择
            node = self._selection(root)
            
            # 2. 扩展
            node = self._expansion(node, balls, my_targets, table)
            
            # 3. 模拟（可能使用2步搜索）
            reward = self._simulation(node, balls, my_targets, table, opponent_targets)
            
            # 4. 反向传播
            self._backpropagation(node, reward)
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"[MCTS] 迭代 {iteration + 1}/{self.num_iterations}")
        
        # 选择访问次数最多的子节点对应的动作
        best_child = max(
            root.children.values(), 
            key=lambda c: c.visits
        ) if root.children else None
        
        if best_child is None:
            if self.verbose:
                print("[MCTS] 未找到有效动作，返回随机动作")
            return self._random_action()
        
        best_action = best_child.action
        if self.verbose:
            avg_reward = best_child.value / best_child.visits if best_child.visits > 0 else 0
            print(f"[MCTS] 最优动作: V0={best_action['V0']:.2f}, "
                  f"phi={best_action['phi']:.2f}, 访问={best_child.visits}, "
                  f"平均奖励={avg_reward:.2f}")
        
        return best_action
    
    def decision(self, balls=None, my_targets=None, table=None, opponent_targets=None):
        """
        Agent决策接口
        
        参数：
            balls: 球状态
            my_targets: 目标球
            table: 球桌
            opponent_targets: 对手的目标球（可选，用于2步搜索）
        
        返回：dict, 击球动作
        """
        if balls is None or my_targets is None or table is None:
            print("[MCTSAgent] 缺少必要参数，返回随机动作")
            return self._random_action()
        
        try:
            remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining) == 0:
                my_targets = ["8"]
            
            # 传递对手目标球信息（如果有）
            return self.search(balls, my_targets, table, opponent_targets)
        
        except Exception as e:
            print(f"[MCTSAgent] 决策失败: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()


# ============================================