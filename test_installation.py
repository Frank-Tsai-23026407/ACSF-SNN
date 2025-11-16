#!/usr/bin/env python3
"""
ACSF-SNN 環境安裝驗證腳本
測試所有必要的依賴和功能是否正常
"""

print('=' * 60)
print('ACSF-SNN 環境驗證測試')
print('=' * 60)
print()

# Test 1: Basic imports
print('[1/5] 測試基本套件導入...')
import torch
import numpy as np
import gym_compat
import gym
import mujoco_py
import spikingjelly
print(f'  ✓ PyTorch {torch.__version__}')
print(f'  ✓ NumPy {np.__version__}')
print(f'  ✓ Gym {gym.__version__}')
print(f'  ✓ MuJoCo-py {mujoco_py.__version__}')
print(f'  ✓ SpikingJelly (0.0.0.0.14)')
print()

# Test 2: CUDA
print('[2/5] 測試 CUDA 可用性...')
print(f'  ✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  ✓ CUDA version: {torch.version.cuda}')
    print(f'  ✓ GPU count: {torch.cuda.device_count()}')
    print(f'  ✓ GPU name: {torch.cuda.get_device_name(0)}')
print()

# Test 3: Gym compatibility
print('[3/5] 測試 Gym 兼容層...')
env = gym.make('CartPole-v1')
state = env.reset()
print(f'  ✓ reset() 回傳類型: {type(state).__name__}')
print(f'  ✓ 是否為 tuple: {isinstance(state, tuple)}')
assert not isinstance(state, tuple), 'Gym compatibility layer failed!'
next_state, reward, done, info = env.step(0)
print(f'  ✓ step() 回傳 4-tuple')
env.close()
print()

# Test 4: MuJoCo environment
print('[4/5] 測試 MuJoCo 環境...')
env = gym.make('Ant-v3')
state = env.reset()
print(f'  ✓ Ant-v3 環境創建成功')
print(f'  ✓ 狀態維度: {state.shape}')
print(f'  ✓ 動作空間: {env.action_space}')
env.close()
print()

# Test 5: Algorithms
print('[5/5] 測試算法模組導入...')
from algorithms import TD3, DDPG, SpikingBCQ, BCQ_AEAD
print('  ✓ TD3')
print('  ✓ DDPG')
print('  ✓ SpikingBCQ')
print('  ✓ BCQ_AEAD (ACSF)')
print()

print('=' * 60)
print('✓ 所有測試通過！環境配置完成。')
print('=' * 60)
print()
print('接下來你可以開始訓練：')
print()
print('1. 訓練行為策略 (TD3):')
print('   python main.py --env=Ant-v3 --seed=9853 --gpu=0 --train_behavioral --mode=TD3')
print()
print('2. 生成重放緩衝區:')
print('   python main.py --env=Ant-v3 --seed=9853 --gpu=0 --generate_buffer --mode=TD3')
print()
print('3. 訓練 SNN (ACSF):')
print('   python main.py --env=Ant-v3 --seed=9853 --gpu=0 --mode=AEAD --buffer=TD3 --T=4')
print()
