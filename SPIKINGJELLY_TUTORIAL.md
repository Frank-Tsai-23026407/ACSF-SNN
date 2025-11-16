# 如何使用 SpikingJelly 框架

`spikingjelly` 是一個基於 PyTorch 的開源脈衝神經網路 (SNN) 框架，旨在讓研究人員和開發者能夠輕鬆地建立、訓練和部署 SNN。本文件將引導您了解其核心概念並提供一個完整的實作範例。

## 1. 安裝

您可以透過 pip 直接安裝 `spikingjelly`。建議同時安裝 PyTorch。

```bash
pip install torch torchvision torchaudio
pip install spikingjelly
```

## 2. 核心概念與元件

`spikingjelly` 的設計與 PyTorch 高度整合，其核心元件可以像 `torch.nn` 中的模組一樣使用。

### 2.1 脈衝神經元 (Neuron)

這是 SNN 的基本建構單元。`spikingjelly` 在 `spikingjelly.activation_based.neuron` 模組中提供了多種神經元模型。最常用的是 **LIF (Leaky Integrate-and-Fire)** 神經元。

*   **運作方式**：神經元接收輸入電流，使其膜電位 (membrane potential) 上升。如果沒有持續輸入，膜電位會隨時間洩漏 (leaky)。當膜電位超過一個閾值 (`v_threshold`) 時，神經元會發出一個脈衝 (spike)，然後膜電位重置。

**程式碼範例：**

```python
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron

# 建立一個 LIF 神經元層
# tau: 膜電位時間常數，控制洩漏速度
# v_threshold: 點火閾值
lif_layer = neuron.LIFNode(tau=2.0, v_threshold=1.0)

# 假設我們有一個 4x8 的輸入張量 (N, C)
x = torch.rand(4, 8)

# LIF 層是時間相關的，我們需要模擬 T 個時間步
T = 10
for t in range(T):
    # 在第 t 步，LIF 層接收輸入 x，並輸出脈衝
    spike_output = lif_layer(x)
    print(f'Time-step {t}: Output shape: {spike_output.shape}')

# 重要的：在每個樣本(sample)開始前，需要重置神經元的狀態（例如膜電位）
lif_layer.reset()
```

### 2.2 編碼器 (Encoder)

真實世界的資料（如圖片）通常是靜態的。為了讓 SNN 處理，我們需要將這些靜態資料轉換為時間序列的脈衝信號。這就是編碼器的作用。

`spikingjelly.encoding` 提供了多種編碼方式，其中 **泊松編碼 (Poisson Encoding)** 很常見，它會根據輸入數值的大小，以一定的機率產生脈衝。

**程式碼範例：**

```python
from spikingjelly.encoding import PoissonEncoder

# 建立一個泊松編碼器，模擬 T=20 個時間步
T = 20
encoder = PoissonEncoder(T)

# 假設有一張 28x28 的灰階圖片 (數值在 0-1 之間)
static_image = torch.rand(1, 28, 28)

# 進行編碼
spike_train = encoder(static_image)

# 輸出是一個時間序列，維度為 (T, N, C, H, W)
print(f'Encoded spike train shape: {spike_train.shape}') # (20, 1, 28, 28)
```

### 2.3 建立一個簡單的 SNN

您可以像搭建標準 PyTorch 模型一樣，將 `spikingjelly` 的神經元層與 `torch.nn` 的層（如 `Linear`, `Conv2d`）結合起來。

**程式碼範例：**

```python
from spikingjelly.activation_based import layer, functional

class SimpleSNN(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.T = T

        # 定義網路結構
        self.net = nn.Sequential(
            layer.Flatten(),
            nn.Linear(28 * 28, 10),
            neuron.LIFNode(tau=2.0)
        )

    def forward(self, x: torch.Tensor):
        # 1. 在每個樣本開始前，重置整個網路的狀態
        functional.reset_net(self)
        
        # 2. 準備一個張量來累積輸出的脈衝總數
        # 輸出層有 10 個神經元
        output_spikes_sum = torch.zeros(x.shape[0], 10).to(x.device)

        # 3. 模擬 T 個時間步
        for t in range(self.T):
            # 輸入 x 的維度是 (T, N, ...)，我們取第 t 步的輸入
            input_t = x[t]
            
            # 執行網路
            output_t = self.net(input_t)
            
            # 累積輸出脈衝
            output_spikes_sum += output_t

        # 4. 返回在 T 個時間步內，每個輸出神經元的總脈衝數
        return output_spikes_sum
```

## 3. 訓練 SNN

由於脈衝函數是不可微分的，`spikingjelly` 使用**替代梯度 (Surrogate Gradient)** 的方法來進行反向傳播。好消息是，這一切都已經在框架內部處理好了，您只需要像平常一樣呼叫 `loss.backward()` 即可。

以下是一個使用上述 `SimpleSNN` 進行訓練的簡化範例：

```python
# --- 1. 準備 ---
T = 100 # 模擬時間步
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. 建立模型和編碼器 ---
model = SimpleSNN(T=T).to(device)
encoder = PoissonEncoder(T)

# --- 3. 準備資料、優化器和損失函數 ---
# 假設我們有 dataloader
# train_loader = ...

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# --- 4. 訓練迴圈 ---
for epoch in range(10):
    # for images, labels in train_loader:
    # --- 模擬一個批次的資料 ---
    images = torch.rand(4, 1, 28, 28) # (N, C, H, W)
    labels = torch.randint(0, 10, (4,))
    # -------------------------
    
    images, labels = images.to(device), labels.to(device)
    
    optimizer.zero_grad()
    
    # a. 將靜態圖片編碼成脈衝序列
    spike_train = encoder(images).to(device) # shape: (T, N, C, H, W)
    
    # b. 將脈衝序列輸入模型
    # 模型的輸出是 T 步內的脈衝總和
    output_spike_sum = model(spike_train)
    
    # c. 計算損失
    # 我們用脈衝總數作為 logits 來計算分類損失
    loss = loss_fn(output_spike_sum, labels)
    
    # d. 反向傳播和更新
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
```

## 4. 總結

`spikingjelly` 透過將 SNN 的元件（神經元、層）封裝成類似 PyTorch 的模組，並自動處理替代梯度，極大地簡化了 SNN 的開發流程。核心步驟包括：

1.  **編碼**：將靜態輸入轉換為時間序列脈衝。
2.  **模型建立**：使用 `spikingjelly.activation_based` 中的元件與 `torch.nn` 混合搭建網路。
3.  **時間步模擬**：在 `forward` 函數中，迴圈模擬 `T` 個時間步，並在開始前使用 `functional.reset_net` 重置網路狀態。
4.  **訓練**：累積多個時間步的輸出脈衝，並基於此計算損失，然後正常進行反向傳播。

若需更進階的功能，如 ANN-to-SNN 轉換、更複雜的神經元模型等，請參考 官方文件。

---

## 5. 進階：使用 STDP 進行無監督學習

除了使用替代梯度進行的反向傳播（監督學習），`spikingjelly` 也支援生物啟發的**局部學習規則 (Local Learning Rules)**，其中最著名的就是 **STDP (Spike-Timing-Dependent Plasticity)**。

### 5.1 STDP 核心概念

STDP 與反向傳播有根本性的不同：

- **局部性 vs. 全域性**: STDP 是一種**局部規則**。權重的更新只依賴於與其直接相連的突觸前神經元和突觸後神經元的脈衝時間差，而不需要一個全域的損失函數。
- **無監督 vs. 有監督**: 基本的 STDP 是一種**無監督學習**機制，它根據輸入脈衝的時序模式來強化或削弱突觸，從而學習輸入資料的特徵。
- **訓練流程**: STDP 的訓練不使用 `loss.backward()` 或 `torch.optim` 優化器。它有自己的 `step()` 函數來更新權重。

`spikingjelly` 在 `spikingjelly.learning` 模組中提供了 STDP 的實作。

### 5.2 STDP 實作範例

以下範例展示如何使用 STDP 訓練一個單層網路來學習輸入模式。

```python
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, layer
from spikingjelly.learning import STDP
from spikingjelly.encoding import PoissonEncoder

# --- 1. 準備 ---
T = 50
N = 1 # Batch size
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. 建立網路和 STDP 學習器 ---
class STDPNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定義網路結構
        self.layer1 = nn.Linear(100, 20) # 輸入層 -> 隱藏層
        self.lif1 = neuron.LIFNode()

        # 建立 STDP 學習器
        # trace_pre: 突觸前脈衝的蹤跡
        # trace_post: 突觸後脈衝的蹤跡
        # a_pre, a_post: 學習率
        self.stdp_learner = STDP(
            synapse=self.layer1, 
            sn=self.lif1, 
            tau_pre=10., 
            tau_post=10.,
            a_pre=0.1,
            a_post=0.1
        )

    def forward(self, x: torch.Tensor):
        # 重置網路和學習器狀態
        functional.reset_net(self)
        self.stdp_learner.reset()

        # 模擬 T 個時間步
        for t in range(self.T):
            input_t = x[t]
            
            # 突觸前脈衝 (這裡直接是輸入)
            pre_spike = input_t
            
            # 網路正向傳播
            post_spike = self.lif1(self.layer1(pre_spike))
            
            # STDP 學習
            # step 函數會根據 pre_spike 和 post_spike 自動更新 self.layer1 的權重
            self.stdp_learner.step(pre_spike, post_spike, on_grad=False)

# --- 3. 訓練迴圈 ---
model = STDPNet().to(device)
model.T = T

# 模擬一個有特定模式的輸入資料 (例如，前 50 個神經元更活躍)
static_input = torch.zeros(N, 100)
static_input[:, :50] = 0.8 

encoder = PoissonEncoder(T)
spike_train = encoder(static_input).to(device) # shape: (T, N, 100)

print("權重更新前 (部分):", model.layer1.weight.data[0, :5])

# 訓練（只需將資料送入網路即可，學習在 forward 中完成）
num_epochs = 5
for i in range(num_epochs):
    model(spike_train)
    print(f"Epoch {i+1} 完成，STDP 已更新權重。")

print("權重更新後 (部分):", model.layer1.weight.data[0, :5])
```

### 5.3 R-STDP (Reward-modulated STDP)

R-STDP 是 STDP 的一種變體，它引入了一個全域的**獎勵 (reward)** 信號來調節權重的更新。這使得 STDP 可以被應用於強化學習任務中。

- **運作方式**: 只有當系統收到正獎勵時，STDP 的權重更新才會被「確認」並應用；收到負獎勵時，更新可能會被抑制或反轉。
- **`spikingjelly.learning.RSTDP`**: `spikingjelly` 提供了 `RSTDP` 學習器，其 `step` 函數需要額外傳入一個 `reward` 參數。

**程式碼修改範例：**

```python
# from spikingjelly.learning import RSTDP

# # 建立 RSTDP 學習器
# rstdp_learner = RSTDP(
#     synapse=self.layer1, 
#     sn=self.lif1,
#     # ... 其他參數 ...
# )

# # 在 forward 迴圈中
# for t in range(self.T):
#     # ... (與 STDP 相同)
#     pre_spike = ...
#     post_spike = ...

#     # 假設我們在每個時間步都能得到一個獎勵信號
#     reward = get_reward_from_environment() # 例如: 1.0 或 -1.0

#     # R-STDP 學習，需要傳入 reward
#     rstdp_learner.step(pre_spike, post_spike, reward, on_grad=False)
```

這個章節展示了 `spikingjelly` 不僅支援主流的基於梯度的訓練方法，也為研究和應用生物啟發的局部學習規則提供了便利的工具。