"""
evaluate_with_log.py - Agent 评估脚本（带日志记录）

功能：
- 让两个 Agent 进行多局对战
- 统计胜负和得分
- 支持切换先后手和球型分配
- 记录每局对战详情和最终结果到日志文件

使用方式：
1. 修改 agent_b 为你设计的待测试的 Agent， 与课程提供的BasicAgent对打
2. 调整 n_games 设置对战局数（评分时设置为120局来计算胜率）
3. 运行脚本查看结果，日志将保存在 evaluate_{时间戳}.log
"""

import os
from datetime import datetime
from utils import set_random_seed
from poolenv import PoolEnv
from agent import BasicAgent, NewAgent, MCTSAgent
import time


class EvaluationLogger:
    """评估日志记录器"""
    
    def __init__(self, agent_a, agent_b, n_games):
        """初始化日志记录器
        
        Args:
            agent_a_name: Agent A 的名称
            agent_b_name: Agent B 的名称
            n_games: 总局数
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"eval_logs/evaluate_{timestamp}.log"
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.agent_a_name = agent_a.__class__.__name__
        self.agent_b_name = agent_b.__class__.__name__
        self.n_games = n_games
        
        os.makedirs("eval_logs", exist_ok=True)
        # 初始化日志文件
        self._write_header()
    
    def _write_header(self):
        """写入日志文件头部信息"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("                    Pool Game Evaluation Log\n")
            f.write("=" * 80 + "\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Agent A: {self.agent_a_name}\n")
            if isinstance(self.agent_a, MCTSAgent):
                f.write(f"Agent A 配置: num_iterations={self.agent_a.num_iterations}, use_2step={self.agent_a.use_2step}, use_defense={self.agent_a.use_defense}\n")
            f.write(f"Agent B: {self.agent_b_name}\n")
            if isinstance(self.agent_b, MCTSAgent):
                f.write(f"Agent B 配置: num_iterations={self.agent_b.num_iterations}, use_2step={self.agent_b.use_2step}, use_defense={self.agent_b.use_defense}\n")
            f.write(f"总局数: {self.n_games}\n")
            f.write("=" * 80 + "\n\n")
    
    def log_game_start(self, game_number, player_a_name, target_ball):
        """记录单局游戏开始
        
        Args:
            game_number: 游戏局数（从0开始）
            player_a_name: 本局Player A的Agent名称
            target_ball: 目标球型
        """
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"第 {game_number + 1} 局 (共 {self.n_games} 局)\n")
            f.write(f"{'='*60}\n")
            f.write(f"Player A: {player_a_name}\n")
            f.write(f"目标球型: {target_ball}\n")
            f.write("-" * 60 + "\n")
    
    def log_game_result(self, game_number, winner, agent_a_result, agent_b_result, hit_count, duration=None):
        """记录单局游戏结果
        
        Args:
            game_number: 游戏局数（从0开始）
            winner: 获胜者 ('A', 'B', 或 'SAME')
            agent_a_result: Agent A 的结果 ('WIN', 'LOSE', 或 'DRAW')
            agent_b_result: Agent B 的结果 ('WIN', 'LOSE', 或 'DRAW')
        """
        # 新增参数: duration (秒) 约定由调用方传入为最后一项
        # 兼容性：如果调用方未传入 duration，这里不报错（保持原样）
        # duration 参数（秒）可选，调用方在记录时传入

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n第 {game_number + 1} 局结果:\n")
            f.write(f"  击球次数: {hit_count}\n")
            if isinstance(duration, (int, float)):
                f.write(f"  时长: {duration:.2f} 秒\n")
            f.write(f"  获胜方: {winner}\n")
            f.write(f"  {self.agent_a_name}: {agent_a_result}\n")
            f.write(f"  {self.agent_b_name}: {agent_b_result}\n")
            f.write("\n")
    
    def log_final_results(self, results):
        """记录最终统计结果
        
        Args:
            results: 结果字典
        """
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("                         最终统计结果\n")
            f.write("=" * 80 + "\n")
            f.write(f"{self.agent_a_name} 获胜: {results['AGENT_A_WIN']} 局\n")
            f.write(f"{self.agent_b_name} 获胜: {results['AGENT_B_WIN']} 局\n")
            f.write(f"平局: {results['SAME']} 局\n")
            f.write("-" * 80 + "\n")
            f.write(f"{self.agent_a_name} 得分: {results['AGENT_A_SCORE']:.1f}\n")
            f.write(f"{self.agent_b_name} 得分: {results['AGENT_B_SCORE']:.1f}\n")
            f.write("-" * 80 + "\n")
            
            # 计算胜率
            total_decisive_games = results['AGENT_A_WIN'] + results['AGENT_B_WIN']
            if total_decisive_games > 0:
                a_win_rate = results['AGENT_A_WIN'] / total_decisive_games * 100
                b_win_rate = results['AGENT_B_WIN'] / total_decisive_games * 100
                f.write(f"{self.agent_a_name} 胜率: {a_win_rate:.2f}% (在 {total_decisive_games} 局决出胜负的对局中)\n")
                f.write(f"{self.agent_b_name} 胜率: {b_win_rate:.2f}% (在 {total_decisive_games} 局决出胜负的对局中)\n")
            
            # 如果提供了平均局时长，记录它
            avg_time = results.get('AVG_GAME_TIME')
            if isinstance(avg_time, (int, float)):
                f.write("-" * 80 + "\n")
                f.write(f"每局平均时长: {avg_time:.2f} 秒\n")
                f.write("-" * 80 + "\n")

            f.write("=" * 80 + "\n")
            f.write(f"日志文件: {self.log_file}\n")
            f.write(f"评估完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
    
    def log_exception(self, game_number, exception):
        """记录异常信息
        
        Args:
            game_number: 游戏局数（从0开始）
            exception: 异常对象
        """
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n⚠️ 第 {game_number + 1} 局发生异常: {exception}\n\n")
    
    def get_log_file_path(self):
        """获取日志文件路径"""
        return os.path.abspath(self.log_file)


def main():
    """主评估函数"""
    # 设置随机种子，enable=True 时使用固定种子，enable=False 时使用完全随机
    set_random_seed(enable=False, seed=42)

    # 初始化环境和Agents
    env = PoolEnv()
    n_games = 20  # 对战局数 自己测试时可以修改 扩充为120局为了减少随机带来的扰动

    agent_a, agent_b = MCTSAgent(num_iterations=20, use_2step=False, use_defense=False), MCTSAgent(num_iterations=20, use_2step=False, use_defense=False)
    
    # 初始化日志记录器
    logger = EvaluationLogger(
        agent_a=agent_a,
        agent_b=agent_b,
        n_games=n_games
    )
    
    # 初始化结果统计
    results = {'AGENT_A_WIN': 0.0, 'AGENT_B_WIN': 0.0, 'SAME': 0.0, 'AVG_GAME_TIME': 0.0, 'AGENT_A_SCORE': 0.0, 'AGENT_B_SCORE': 0.0}
    # 记录每局时长（秒）
    game_times = []
    
    players = [agent_a, agent_b]  # 用于切换先后手
    target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']  # 轮换球型

    # 开始对战循环
    try:
        for i in range(n_games): 
            print()
            print(f"------- 第 {i + 1} 局比赛开始 -------")
            
            env.reset(target_ball=target_ball_choice[i % 4])
            player_class = players[i % 2].__class__.__name__
            ball_type = target_ball_choice[i % 4]
            
            print(f"本局 Player A: {player_class}, 目标球型: {ball_type}")
            
            # 记录游戏开始
            logger.log_game_start(i, player_class, ball_type)
            # 记录本局开始时间
            start_time = time.time()
            
            # 游戏循环
            while True:
                player = env.get_curr_player()
                print(f"[第{env.hit_count}次击球] player: {player}")
                obs = env.get_observation(player)
                
                if player == 'A':
                    action = players[i % 2].decision(*obs)
                else:
                    action = players[(i + 1) % 2].decision(*obs)
                
                step_info = env.take_shot(action)
                
                done, info = env.get_done()
                if not done:
                    if step_info.get('ENEMY_INTO_POCKET'):
                        print(f"对方球入袋：{step_info['ENEMY_INTO_POCKET']}")
                
                if done:
                    # 统计结果（player A/B 转换为 agent A/B）
                    winner = info['winner']
                    
                    if winner == 'SAME':
                        results['SAME'] += 1
                        agent_a_result = 'DRAW'
                        agent_b_result = 'DRAW'
                    elif winner == 'A':
                        # Player A 获胜
                        if i % 2 == 0:  # Agent A 是 Player A
                            results['AGENT_A_WIN'] += 1
                            agent_a_result = 'WIN'
                            agent_b_result = 'LOSE'
                        else:  # Agent B 是 Player A
                            results['AGENT_B_WIN'] += 1
                            agent_a_result = 'LOSE'
                            agent_b_result = 'WIN'
                    else:  # winner == 'B'
                        # Player B 获胜
                        if i % 2 == 0:  # Agent B 是 Player B
                            results['AGENT_B_WIN'] += 1
                            agent_a_result = 'LOSE'
                            agent_b_result = 'WIN'
                        else:  # Agent A 是 Player B
                            results['AGENT_A_WIN'] += 1
                            agent_a_result = 'WIN'
                            agent_b_result = 'LOSE'
                    
                    # 记录本局时长并保存结果
                    duration = time.time() - start_time
                    game_times.append(duration)

                    # 记录游戏结果（含时长）
                    logger.log_game_result(i, winner, agent_a_result, agent_b_result, env.hit_count, duration=duration)
                    print(f"本局结果：{winner} 获胜 | {agent_a.__class__.__name__}: {agent_a_result}, {agent_b.__class__.__name__}: {agent_b_result}, 击球次数: {env.hit_count}, 时长: {duration:.2f}s")
                    break
    except Exception as e:
        print(f"评估过程中发生异常: {e}")
        logger.log_exception(i, e)

    # 计算分数：胜1分，负0分，平局0.5
    results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] * 1 + results['SAME'] * 0.5
    results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] * 1 + results['SAME'] * 0.5

    # 记录最终结果
    # 计算并写入平均每局时长
    if len(game_times) > 0:
        avg_time = sum(game_times) / len(game_times)
        results['AVG_GAME_TIME'] = avg_time
    else:
        results['AVG_GAME_TIME'] = 0.0
    logger.log_final_results(results)

    # 输出最终结果
    print("\n" + "=" * 80)
    print("最终结果：", results)
    print("=" * 80)
    print(f"日志已保存到: {logger.get_log_file_path()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
