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


class MCTSAgent(Agent):
    """基于蒙特卡洛树搜索的 Agent"""
    
    def __init__(self, num_iterations=50, exploration_c=1.414, 
                 use_heuristic=True, verbose=False):
        """
        参数：
            num_iterations: MCTS迭代次数
            exploration_c: UCB1探索系数
            use_heuristic: 是否使用启发式生成动作候选
            verbose: 是否打印调试信息
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.exploration_c = exploration_c
        self.use_heuristic = use_heuristic
        self.verbose = verbose
        self.new_agent = NewAgent()  # 用于启发式候选和快速模拟
    
    def _generate_action_candidates(self, balls, my_targets, table, num_samples=10):
        """
        生成动作候选集合
        
        结合启发式和随机采样：
        1. 用NewAgent的启发式生成高质量候选
        2. 添加随机采样作为多样性
        
        返回：list of dicts，每个dict是一个动作
        """
        candidates = []
        
        # 方式1: 如果启用启发式，先用NewAgent生成候选
        if self.use_heuristic:
            from utils import vec2, norm, angle_deg, seg_dist
            
            R = 0.028575
            cue_pos = vec2(balls["cue"].state.rvw[0])
            pockets = list(table.pockets.values())
            
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
                    
                    # 简化：不检查阻挡，加速候选生成
                    phi = angle_deg(ghost - cue_pos)
                    d = np.linalg.norm(ghost - cue_pos)
                    V0 = float(np.clip(1.2 + 0.8 * d, 0.5, 6.0))
                    
                    candidates.append({
                        'V0': V0, 'phi': phi, 'theta': 0.0, 
                        'a': 0.0, 'b': 0.0
                    })
        
        # 方式2: 添加随机采样（保证多样性）
        num_random = max(1, num_samples - len(candidates))
        for _ in range(num_random):
            candidates.append(self._random_action())
        
        # 返回前num_samples个候选
        return candidates[:num_samples]
    
    def _action_to_key(self, action):
        """将动作转换为字典键（便于在字典中查找）"""
        return tuple(action[k] for k in ['V0', 'phi', 'theta', 'a', 'b'])
    
    def _simulate_action(self, balls, table, action, my_targets):
        """
        快速模拟一个击球动作，返回奖励
        
        返回：float, 该动作的奖励得分
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
            
            # 计算奖励
            score = analyze_shot_for_reward(shot, last_state, my_targets)
            return score
            
        except Exception as e:
            if self.verbose:
                print(f"[MCTS] 模拟失败: {e}")
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
    
    def _simulation(self, node, balls, my_targets, table):
        """
        模拟阶段：快速评估该节点的价值
        
        这里我们直接模拟该节点的动作，得到立即奖励
        （简化版：不进行深层前向搜索）
        
        返回：float, 该节点的奖励值
        """
        if node.action is None:
            # 根节点，不执行任何动作
            return 0
        
        # 模拟该动作
        reward = self._simulate_action(balls, table, node.action, my_targets)
        return reward
    
    def _backpropagation(self, node, reward):
        """
        反向传播阶段：更新节点的访问次数和累积奖励
        """
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def search(self, balls, my_targets, table):
        """
        执行MCTS搜索，返回最优动作
        
        参数：
            balls: 球状态字典
            my_targets: 目标球列表
            table: 球桌对象
        
        返回：dict, 最优击球动作
        """
        if self.verbose:
            print(f"[MCTS] 开始搜索 (iterations={self.num_iterations})")
        
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
            
            # 3. 模拟
            reward = self._simulation(node, balls, my_targets, table)
            
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
    
    def decision(self, balls=None, my_targets=None, table=None):
        """
        Agent决策接口
        
        参数：
            balls: 球状态
            my_targets: 目标球
            table: 球桌
        
        返回：dict, 击球动作
        """
        if balls is None or my_targets is None or table is None:
            print("[MCTSAgent] 缺少必要参数，返回随机动作")
            return self._random_action()
        
        try:
            remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining) == 0:
                my_targets = ["8"]
            
            return self.search(balls, my_targets, table)
        
        except Exception as e:
            print(f"[MCTSAgent] 决策失败: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()


# ============================================