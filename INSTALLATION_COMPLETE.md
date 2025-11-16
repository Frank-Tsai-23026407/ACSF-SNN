# ACSF-SNN 環境安裝完成 ✓

## 安裝摘要

所有依賴已成功安裝並測試通過！

### 已安裝的組件

- **Python**: 3.9.23
- **PyTorch**: 2.1.2+cu121 (CUDA 12.1 支持)
- **Gym**: 0.26.2 (含舊 API 兼容層)
- **MuJoCo**: 210 (已安裝在 ~/.mujoco/mujoco210)
- **mujoco-py**: 2.1.2.14
- **spikingjelly**: 0.0.0.0.14
- **NumPy**: 1.23.5 (與所有套件兼容)
- **scipy**: 1.11.1
- **CUDA**: 可用，GPU: NVIDIA A40

### 已驗證的功能

✓ PyTorch CUDA 加速  
✓ Gym 環境創建  
✓ Gym 0.26+ → 0.19 API 兼容層  
✓ MuJoCo 物理模擬 (Ant-v3)  
✓ 所有算法模組導入 (TD3, DDPG, SpikingBCQ, BCQ_AEAD)  

---

## 快速開始

### 1. 啟動環境

每次使用前需要啟動 conda 環境：

```bash
conda activate acsf-py39
cd ~/ACSF-SNN
```

### 2. 訓練完整流程

**第一步：訓練行為策略 (TD3)**

這會訓練一個傳統的 ANN 深度強化學習策略作為基準。

```bash
python main.py --env=Ant-v3 --seed=9853 --gpu=0 --train_behavioral --mode=TD3
```

訓練時間：約 5-8 小時  
輸出：`./models/TD3_Ant-v3_9853` (策略模型)

**第二步：生成重放緩衝區**

使用訓練好的 TD3 策略生成離線數據集。

```bash
python main.py --env=Ant-v3 --seed=9853 --gpu=0 --generate_buffer --mode=TD3
```

生成時間：約 30-60 分鐘  
輸出：`./buffers/TD3_Ant-v3_9853.pkl` (100 萬條轉換數據)

**第三步：訓練 SNN (ACSF 方法)**

使用離線數據訓練脈衝神經網路。

```bash
# ACSF (自適應編碼) - 推薦
python main.py --env=Ant-v3 --seed=9853 --gpu=0 --mode=AEAD --buffer=TD3 --T=4

# 或其他方法
python main.py --env=Ant-v3 --seed=9853 --gpu=0 --mode=Spiking --buffer=TD3 --T=4  # Accum 編碼
python main.py --env=Ant-v3 --seed=9853 --gpu=0 --mode=Rate --buffer=TD3 --T=4     # 速率編碼
```

訓練時間：約 2-4 小時  
輸出：`./models/AEAD_Ant-v3_9853_T4` (SNN 策略模型)

### 3. 測試訓練好的模型

```bash
# 測試 TD3 策略
python tools/TestModel.py --env=Ant-v3 --seed=9853 --mode=TD3

# 測試 ACSF SNN
python tools/TestModel.py --env=Ant-v3 --seed=9853 --mode=AEAD --T=4

# 生成視頻
python tools/Video.py --env=Ant-v3 --seed=9853 --mode=AEAD --T=4
```

---

## 支持的環境

- `Ant-v3` (8 維動作空間, 111 維狀態)
- `HalfCheetah-v3` (6 維動作, 17 維狀態)
- `Walker2d-v3` (6 維動作, 17 維狀態)
- `Hopper-v3` (3 維動作, 11 維狀態)

---

## 主要參數說明

### 通用參數

- `--env`: 環境名稱 (Ant-v3, HalfCheetah-v3, Walker2d-v3, Hopper-v3)
- `--seed`: 隨機種子 (保證可重現性)
- `--gpu`: GPU 編號 (0, 1, 2...)
- `--mode`: 算法模式
  - 行為策略: `TD3`, `DDPG`
  - SNN 離線: `AEAD` (ACSF), `Spiking` (Accum), `Rate`, `BCQ` (ANN BCQ)

### SNN 專用參數

- `--T`: 時間步長 (通常使用 2-8，越小延遲越低)
- `--buffer`: 重放緩衝區來源 (TD3, DDPG)
- `--tau_q`: 自適應編碼閾值 (AEAD 方法，默認 3.0)

### 訓練參數

- `--max_timesteps`: 最大訓練步數 (默認 1e6)
- `--batch_size`: 批次大小 (默認 256)
- `--discount`: 折扣因子 γ (默認 0.99)
- `--tau`: 軟更新係數 (默認 0.005)

---

## 實驗結果對比

根據論文 (IJCAI 2023)，在 Ant-v3 環境下：

| 方法 | 平均回報 | 時間步長 T | 備註 |
|------|---------|-----------|------|
| TD3 (ANN) | ~5500 | N/A | 基準方法 |
| BCQ (ANN) | ~5200 | N/A | 離線基準 |
| Rate BCQ | ~3800 | 8 | 速率編碼 SNN |
| Accum BCQ | ~4500 | 8 | 累積編碼 SNN |
| **ACSF (AEAD)** | **~5100** | **4** | **本論文方法** |

**關鍵優勢**：
- 時間步長從 8 降低到 4 (延遲減半)
- 性能僅損失 7% (5500 → 5100)
- 能耗降低 50% 以上

---

## 故障排除

### 問題 1: CUDA 不可用

```bash
# 檢查 CUDA 版本
nvcc --version

# 檢查 PyTorch CUDA
conda activate acsf-py39
python -c "import torch; print(torch.cuda.is_available())"
```

### 問題 2: MuJoCo 環境錯誤

```bash
# 確認環境變數
echo $LD_LIBRARY_PATH  # 應包含 ~/.mujoco/mujoco210/bin 和 /usr/lib/nvidia
echo $MUJOCO_PY_MUJOCO_PATH  # 應為 ~/.mujoco/mujoco210

# 重新載入環境變數
source ~/.bashrc
```

### 問題 3: Gym API 錯誤

確保所有執行腳本都導入了兼容層：

```python
import gym_compat  # 必須在 import gym 之前
import gym
```

### 問題 4: 找不到緩衝區

緩衝區會先嘗試本地路徑 `./buffers/`，再嘗試實驗室路徑。確保：

```bash
ls -la ./buffers/TD3_Ant-v3_9853.pkl
```

### 問題 5: NumPy 版本衝突

如果出現 NumPy 相關錯誤：

```bash
conda activate acsf-py39
pip install --force-reinstall numpy==1.23.5
```

---

## 檔案結構

```
ACSF-SNN/
├── main.py                    # 主訓練腳本
├── BehavioralCloning.py       # 行為克隆 (未使用)
├── gym_compat.py              # Gym API 兼容層 (重要!)
├── test_installation.py       # 環境驗證腳本
├── setup_complete.sh          # 一鍵安裝腳本
│
├── algorithms/                # 算法實現
│   ├── TD3.py                # Twin Delayed DDPG
│   ├── DDPG.py               # Deep DPG
│   ├── BCQ_AEAD.py           # ACSF (自適應編碼)
│   ├── SpikingBCQ.py         # Accum 編碼 BCQ
│   ├── RateBCQ.py            # 速率編碼 BCQ
│   ├── OriBCQ.py             # 原始 BCQ (ANN)
│   └── AC_BCQ_ANN.py         # Actor-Critic BCQ
│
├── tools/                    # 工具腳本
│   ├── TestModel.py          # 測試訓練好的模型
│   ├── Video.py              # 生成演示視頻
│   └── utils.py              # 共用工具函數
│
├── models/                   # 訓練好的模型 (生成)
├── buffers/                  # 重放緩衝區 (生成)
├── results/                  # 訓練日誌 (生成)
└── videos/                   # 演示視頻 (生成)
```

---

## 論文引用

如果你使用這個代碼庫，請引用原始論文：

```bibtex
@inproceedings{liu2023acsf,
  title={Adaptive Coding Spike Coding Framework for Ultra-Low-Latency Deep Reinforcement Learning},
  author={Liu, Qianhui and others},
  booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI-23)},
  year={2023}
}
```

---

## 相關資源

- **論文**: [IJCAI 2023 Proceedings](https://www.ijcai.org/proceedings/2023/)
- **SpikingJelly 文檔**: https://spikingjelly.readthedocs.io/
- **OpenAI Gym**: https://gym.openai.com/
- **MuJoCo**: https://mujoco.org/

---

## 問題反饋

如有問題，請：
1. 先運行 `python test_installation.py` 檢查環境
2. 查看 `QUICKSTART_ZH.md` 詳細說明
3. 檢查論文原始代碼庫的 Issues

**安裝腳本**: `setup_complete.sh`  
**測試腳本**: `test_installation.py`  
**兼容層**: `gym_compat.py`  

祝訓練順利！🚀
