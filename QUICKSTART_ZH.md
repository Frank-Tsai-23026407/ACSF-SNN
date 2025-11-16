# ACSF-SNN 快速開始指南

## 環境設定（重要！）

由於此專案使用 gym 0.26.2 和 mujoco-py，**必須使用 Python 3.9**（不是 3.11）。

### 一鍵安裝（推薦）

```bash
cd ~/ACSF-SNN
bash setup_complete.sh
```

這個腳本會自動完成：
- ✓ 下載並安裝 MuJoCo 210
- ✓ 建立 Python 3.9 conda 環境 (acsf-py39)
- ✓ 安裝所有必要套件 (PyTorch, gym, mujoco-py, spikingjelly 等)
- ✓ 設定環境變數
- ✓ 測試所有安裝是否成功
- ✓ 建立必要的目錄 (models, buffers, results, videos)

**預計時間：** 10-15 分鐘（視網路速度而定）

安裝完成後：
```bash
conda activate acsf-py39
cd ~/ACSF-SNN
```

### 手動安裝（進階）

如果自動腳本失敗，可以手動執行以下步驟：

<details>
<summary>點擊展開手動安裝步驟</summary>

```bash
# 1. 建立 Python 3.9 環境
conda create -n acsf-py39 python=3.9 -y
conda activate acsf-py39

# 2. 下載並安裝 MuJoCo 210
mkdir -p ~/.mujoco && cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz

# 3. 設定環境變數（加到 ~/.bashrc 永久生效）
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin' >> ~/.bashrc
echo 'export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210' >> ~/.bashrc
source ~/.bashrc

# 4. 安裝所有 Python 套件（使用自動化腳本）
cd ~/ACSF-SNN
bash install_packages.sh
```

### 手動安裝（若自動腳本失敗）

```bash
# 確認在 acsf-py39 環境
conda activate acsf-py39
cd ~/ACSF-SNN

# 1. 安裝 PyTorch（依據你的 CUDA 版本調整）
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 2. 安裝 Cython 和 numpy
pip install 'Cython<3' 'numpy==1.23.5'

# 3. 安裝 gym
pip install 'setuptools<66' wheel
pip install gym==0.26.2

# 4. 安裝 mujoco-py
pip install mujoco-py==2.1.2.14

# 5. 安裝 spikingjelly
pip install spikingjelly==0.0.0.0.14

# 6. 安裝其他相依套件
pip install scipy==1.11.1 tqdm pyyaml matplotlib opencv-python imageio cloudpickle
```

</details>

### 測試安裝

```bash
# 啟動環境
conda activate acsf-py39

# 測試基本套件
python -c "import gym_compat; import gym; import torch; import spikingjelly; print('✓ 基本套件載入成功')"

# 測試 MuJoCo（首次執行會編譯，需要數分鐘）
python -c "import mujoco_py; print('✓ mujoco-py 載入成功')"

# 測試完整環境
python -c "import gym_compat; import gym; env = gym.make('Ant-v3'); state = env.reset(); print('✓ MuJoCo 環境可用，reset 回傳:', type(state))"
```

**重要：** 
- 首次 import mujoco_py 會自動編譯，可能需要 2-5 分鐘
- gym 0.26.2 需要使用 `gym_compat.py` 相容層（已自動加入程式碼）
- `reset()` 現在回傳單一 state（不是 tuple），與舊程式碼相容

**如果編譯失敗，請確認：**
- 已設定 `LD_LIBRARY_PATH` 和 `MUJOCO_PY_MUJOCO_PATH`
- 已安裝系統相依套件：`sudo apt-get install -y libosmesa6-dev patchelf libgl1 libglew-dev`

---

## SNN 訓練流程

### 第一步：訓練行為策略（Behavioral Policy）

選擇 DDPG 或 TD3 作為行為策略：

```bash
conda activate acsf-py39

# 使用 TD3（推薦）
python main.py --env=Ant-v3 --seed=9853 --gpu=0 --train_behavioral --mode=TD3

# 或使用 DDPG
python main.py --env=Ant-v3 --seed=9853 --gpu=0 --train_behavioral --mode=DDPG
```

**說明：**
- `--env`: MuJoCo 環境（Ant-v3, HalfCheetah-v3, Walker2d-v3, Hopper-v3）
- `--seed`: 隨機種子（建議用 9853 對齊預設命名）
- `--gpu`: GPU 編號
- 訓練好的模型會存在 `./models/TD3_Ant-v3_9853`

### 第二步：產生離線資料緩衝區（Replay Buffer）

```bash
# 用訓練好的 TD3 產生 buffer
python main.py --env=Ant-v3 --seed=9853 --gpu=0 --generate_buffer --mode=TD3
```

**產生的檔案：**
- `./buffers/Robust_Ant-v3_9853_{state,action,next_state,reward,not_done,ptr}.npy`

### 第三步：訓練 SNN（離線強化學習）

#### 方法 A: ACSF（論文主方法，可學習編碼器）

```bash
python main.py --env=Ant-v3 --seed=9853 --gpu=0 --mode=AEAD --buffer=TD3 --T=4
```

#### 方法 B: Spiking Baseline（累積編碼 + SNN）

```bash
python main.py --env=Ant-v3 --seed=9853 --gpu=0 --mode=Spiking --buffer=TD3 --T=8
```

**重要參數：**
- `--mode`: 
  - `AEAD`: ACSF 主方法（低延遲可學習編碼）
  - `Spiking`: Accum. coding baseline
  - `Rate`: Rate coding baseline
  - `BCQ`: 原始 BCQ（ANN）
- `--T`: SNN 時間步數（AEAD 預設 4，Spiking 常用 8～16）
- `--buffer`: 使用的 buffer 來源（`TD3`, `DDPG_9853`, `DDPG_0`）

**輸出：**
- 模型: `./models/BCQ_AEAD_Ant-v3_9853_TD3`
- 評估結果: `./results/BCQ_AEAD_Ant-v3_9853_TD3.npy`

---

## 測試與視覺化

### 測試訓練好的模型

```bash
# 測試 AEAD 模型
python tools/TestModel.py --env=Ant-v3 --seed=9853 --mode=AEAD --gpu=0
```

### 產生影片

```bash
# 產生環境互動影片
python tools/Video.py --env=Ant-v3 --seed=9853 --buffer=TD3 --gpu=0
```

影片會存在 `./videos/BCQ_AEAD_Ant-v3_9853_TD3/`

---

## 批次訓練（多環境）

使用提供的 shell 腳本在多個環境同時訓練：

```bash
# 訓練 Spiking baseline（需要編輯腳本內的 conda env 名稱）
bash scripts/TrainAC_SpikingBaseline.sh

# 訓練 AEAD（不同時間步）
bash scripts/TrainAC_TimeSteps.sh
```

**注意：** 腳本使用 tmux 分割視窗並行訓練，需要先安裝 `tmux`。

---

## 常見問題

### 1. MuJoCo 錯誤：找不到 libmujoco210.so

```bash
# 確認 LD_LIBRARY_PATH 已設定
echo $LD_LIBRARY_PATH
# 應該包含 ~/.mujoco/mujoco210/bin 或 /home/你的使用者名稱/.mujoco/mujoco210/bin

# 若未設定，執行：
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210

# 永久設定（加到 ~/.bashrc）
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin' >> ~/.bashrc
echo 'export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210' >> ~/.bashrc
source ~/.bashrc
```

### 2. mujoco-py 編譯錯誤

**問題：** `error: command 'gcc' failed` 或缺少標頭檔

**解決方法：**
```bash
# 安裝必要的系統套件
sudo apt-get update
sudo apt-get install -y build-essential patchelf libosmesa6-dev libgl1 libglew-dev libglfw3
```

### 3. 找不到 replay buffer

錯誤訊息：`Replay buffer not found`

**解決方法：**
- 確認你已經執行過 `--generate_buffer`
- 檔案應該在 `./buffers/Robust_{env}_{seed}_*.npy`
- 或者手動指定 buffer 路徑（修改 `main.py` 第 150 行附近）

### 3. 找不到 replay buffer

錯誤訊息：`Replay buffer not found`

**解決方法：**
- 確認你已經執行過 `--generate_buffer`
- 檔案應該在 `./buffers/Robust_{env}_{seed}_*.npy`
- 檢查檔名是否匹配（例如 `Robust_Ant-v3_9853_state.npy`）
- 程式會自動嘗試本地 `./buffers/` 和實驗室路徑

### 4. CUDA 版本不匹配

如果你的 CUDA 不是 11.8，調整 PyTorch 安裝指令：

```bash
# 檢查 CUDA 版本
nvcc --version

# CUDA 12.1
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.7
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# CPU only（無 GPU）
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

### 5. Python 版本錯誤

**問題：** 使用 Python 3.11 導致 gym 安裝失敗

**解決方法：** 必須使用 Python 3.9。若已建立錯誤版本的環境：
```bash
# 刪除舊環境
conda env remove -n acsf-snn-env

# 建立正確的 Python 3.9 環境
conda create -n acsf-py39 python=3.9 -y
conda activate acsf-py39
```

---

## 檔案結構

```
ACSF-SNN/
├── main.py                 # 主程式（訓練/產生 buffer/離線訓練）
├── algorithms/             # 演算法實作
│   ├── SpikingBCQ.py      # SNN + Accum. coding
│   ├── BCQ_AEAD.py        # ACSF 主方法
│   ├── RateBCQ.py         # Rate coding
│   └── ...
├── tools/                  # 工具
│   ├── TestModel.py       # 測試模型
│   ├── Video.py           # 產生影片
│   └── utils.py           # Replay buffer
├── scripts/                # 批次訓練腳本
├── models/                 # 儲存訓練好的模型
├── buffers/                # 儲存 replay buffer
├── results/                # 儲存評估結果
└── videos/                 # 儲存影片
```

---

## 引用

如果使用此程式碼，請引用原論文：

```bibtex
@inproceedings{ijcai2023p0340,
  title     = {A Low Latency Adaptive Coding Spike Framework for Deep Reinforcement Learning},
  author    = {Qin, Lang and Yan, Rui and Tang, Huajin},
  booktitle = {Proceedings of IJCAI-23},
  pages     = {3049--3057},
  year      = {2023}
}
```
