#!/usr/bin/env python3
"""
test_mcts.py - MCTS框架快速测试脚本

测试MCTSAgent是否能正常工作
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from agent import MCTSAgent, NewAgent, BasicAgent
from poolenv import PoolEnv

def test_mcts_agent():
    """测试MCTS Agent的基本功能"""
    print("=" * 60)
    print("开始测试 MCTSAgent")
    print("=" * 60)
    
    # 初始化环境
    env = PoolEnv()
    env.reset(target_ball='solid')
    
    # 获取初始观测
    balls, my_targets, table = env.get_observation()
    
    print(f"初始状态:")
    print(f"  - 目标球: {my_targets}")
    print(f"  - 球的数量: {len(balls)}")
    print(f"  - 袋口数: {len(table.pockets)}")
    
    # 创建MCTS Agent
    mcts_agent = MCTSAgent(
        num_iterations=30,  # 快速测试，迭代次数较少
        exploration_c=1.414,
        use_heuristic=True,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("MCTS Agent 执行决策...")
    print("=" * 60)
    
    try:
        action = mcts_agent.decision(balls=balls, my_targets=my_targets, table=table)
        
        print("\n" + "=" * 60)
        print("决策成功！")
        print("=" * 60)
        print(f"返回的动作:")
        print(f"  V0 (初速度): {action['V0']:.2f} m/s")
        print(f"  phi (水平角): {action['phi']:.2f}°")
        print(f"  theta (竖直角): {action['theta']:.2f}°")
        print(f"  a (横向偏移): {action['a']:.3f}")
        print(f"  b (纵向偏移): {action['b']:.3f}")
        
        return True
    
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 决策失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


def compare_agents():
    """对比 NewAgent 和 MCTSAgent 的执行时间"""
    print("\n" + "=" * 60)
    print("对比 NewAgent vs MCTSAgent 执行时间")
    print("=" * 60)
    
    import time
    
    env = PoolEnv()
    env.reset(target_ball='solid')
    balls, my_targets, table = env.get_observation()
    
    # 测试 NewAgent
    print("\n1. NewAgent (启发式搜索):")
    new_agent = NewAgent()
    start = time.time()
    action_new = new_agent.decision(balls=balls, my_targets=my_targets, table=table)
    time_new = time.time() - start
    print(f"   执行时间: {time_new:.2f}s")
    print(f"   V0={action_new['V0']:.2f}, phi={action_new['phi']:.2f}")
    
    # 测试 MCTSAgent
    print("\n2. MCTSAgent (蒙特卡洛树搜索, 30次迭代):")
    mcts_agent = MCTSAgent(num_iterations=30, verbose=False)
    start = time.time()
    action_mcts = mcts_agent.decision(balls=balls, my_targets=my_targets, table=table)
    time_mcts = time.time() - start
    print(f"   执行时间: {time_mcts:.2f}s")
    print(f"   V0={action_mcts['V0']:.2f}, phi={action_mcts['phi']:.2f}")
    
    print(f"\n速度对比:")
    print(f"   MCTSAgent / NewAgent = {time_mcts / time_new:.2f}x")
    
    # 测试 BasicAgent
    print("\n3. BasicAgent (贝叶斯优化):")
    basic_agent = BasicAgent()
    start = time.time()
    action_basic = basic_agent.decision(balls=balls, my_targets=my_targets, table=table)
    time_basic = time.time() - start
    print(f"   执行时间: {time_basic:.2f}s")
    print(f"   V0={action_basic['V0']:.2f}, phi={action_basic['phi']:.2f}")


if __name__ == '__main__':
    # 测试基本功能
    success = test_mcts_agent()
    
    if success:
        # 如果基本测试成功，进行性能对比
        try:
            compare_agents()
        except Exception as e:
            print(f"\n性能对比失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
