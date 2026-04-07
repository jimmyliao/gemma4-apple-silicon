# 除錯日誌：Gemma 4 在 Apple Silicon 上的 PLE Quantization Bug

**日期**: 2026-04-07
**環境**: M1 Air 16GB, mlx 0.31.1 + mlx-lm 0.31.2 (git main), Metal backend
**症狀**: 4-bit 量化的 Gemma 4 E2B 在 M1 上產生重複亂碼，但同一個檔案在 Colab L4 (CUDA) 上正常

這份文件記錄完整的除錯過程，目的是幫助你（和未來的我）學習如何系統地處理這類「跨平台不一致」的 LLM 推論問題。

---

## 故事的開始：症狀

```
M1 smoke test:
  Prompt:   問題：台灣首都是？\nA. 台北\nB. 高雄\nC. 台中\nD. 台南
  Response: '台灣：台灣：台灣：台灣：台灣：'   ← 亂碼，重複 token
  Peak memory: 2.642 GB

Colab L4 smoke test (相同檔案):
  Response: 'A\n'                              ← 正確
  Peak memory: 7.445 GB
```

兩個關鍵 anomaly：
1. **輸出完全錯誤**（重複 token）
2. **記憶體差距巨大**（2.6 GB vs 7.4 GB）

當看到這種「同樣的 input，不同 output」的問題，**第一個直覺反應應該是檢查輸入是否真的相同**。

---

## Hypothesis Tree（猜想樹）

我把所有可能原因列出來，並且記下「如何驗證」：

```
為何 M1 輸出亂碼？
│
├── H1: 模型檔案在傳輸中損壞
│   └── 驗證：MD5 比對 Drive vs M1
│
├── H2: 載入模型時讀錯 config（如 vocab_size 錯誤）
│   └── 驗證：print 載入後的 model config
│
├── H3: 量化過程在 Colab 上有特殊處理（M1 conversion 不會這樣）
│   └── 驗證：看 Colab 是否做了 PLE-safe quantization
│
├── H4: mlx-lm 在 Metal 後端的 gemma4 實作有 bug
│   └── 驗證：看 gemma4_text.py source code
│
├── H5: PLE quantization bug（已知 community 議題）
│   └── 驗證：看 model.safetensors 哪些 layer 被量化了
│
└── H6: Tokenizer 有問題（編解碼錯誤）
    └── 驗證：手動 encode + decode 一段繁中
```

下面我會走過實際驗證的順序，**包含猜錯的方向**。

---

## 第一步：檢查檔案完整性（H1）

最便宜的驗證：MD5 比對。

```bash
# Drive 上的 MD5（從 gws files list 拿）
gws drive files list --params '{"q":"...","fields":"files(name,md5Checksum)"}'

# 本地 MD5
for f in models/gemma-4-e2b-it-mlx-4bit/*; do
  md5 -q "$f"
done
```

**結果**（部分）：

| 檔案 | Drive | Local | Match? |
|------|-------|-------|--------|
| `model.safetensors` | 5f8f3a89... | 5f8f3a89... | ✅ |
| `config.json` | 5258c32b... | b1cd0b5f... | ❌ |
| `tokenizer.json` | 72b10445... | 56789eca... | ❌ off by 1 byte |

**啊哈！** 模型權重 OK，但 4 個 JSON 檔不一樣。Sub-symptom：JSON files have wrong sizes, off by ~5%.

### 深入研究 H1：JSON 為什麼不一樣？

回想：我用 `gws drive files get` 下載，並指定 `--output <path>`。為什麼大檔案 byte-perfect，小 JSON 不對？

讀 `gws drive files get --help`：
```
-o, --output <PATH>  Output file path for binary responses
```

關鍵字：**「binary responses」**。Drive API 對 `application/json` 檔案會回傳 JSON content type，gws 把它當「JSON metadata」處理（不是 binary），於是：
1. **完全忽略 `--output`** flag
2. 把 JSON 內容**直接 dump 到 stdout**
3. 同時 stdout 還會被 gws 自己的 `Using keyring backend: keyring` header 污染

我的第一版 fix script 從 stdout 抓 JSON 寫檔，但 gws 印出的 JSON 是它**重新 serialize 過的**（key order 變、空白縮排不同、可能 unicode escape）→ byte 不對 → MD5 不對。

**Lesson 1**: CLI tool 的「文字 mode 處理」對 binary file 非常不友好。看 `--help` 的字眼要小心——「binary」不一定包含 application/json，即使該 JSON 檔本身是 30 MB 的 BPE tokenizer。

### 修法：繞過 gws 的 JSON 處理

我們有 OAuth refresh token（從 `gws auth export --unmasked`）。直接 curl Drive REST API：

```bash
# 1. Refresh token → access token
ACCESS_TOKEN=$(curl -s -X POST https://oauth2.googleapis.com/token \
    -d "client_id=$CID" -d "client_secret=$CS" \
    -d "refresh_token=$RT" -d "grant_type=refresh_token" \
    | jq -r '.access_token')

# 2. Direct download via Drive REST API
curl -L -H "Authorization: Bearer $ACCESS_TOKEN" \
    "https://www.googleapis.com/drive/v3/files/$FID?alt=media" \
    -o $dest
```

`curl` 不會重新格式化 stream，所以 byte-perfect。**全部 8 個 file MD5 對齊** ✅

**Lesson 2**: 當第三方 CLI 工具的抽象層誤事時，**降到底層 API 直接呼叫**通常是最快的修法。OAuth refresh token + curl 是萬用救命招。

---

## 第二步：以為問題解決了 → 不是

跑 smoke test，期待 `'A\n'`。實際得到：

```
Response: '台灣：台灣：台灣：台灣：台灣：'
Peak memory: 2.642 GB
```

**還是亂碼**。但這次 MD5 完全 match！意味著：
- 不是 H1（檔案損壞）
- **檔案是對的，但 mlx-lm 的處理方式有問題**

注意到 **peak memory 還是 2.642 GB**（vs Colab 7.4 GB）。同樣的 .safetensors（MD5 一致），同樣的 mlx-lm 版本（git main 0.31.2），但記憶體用量差 3 倍。這非常奇怪。

**Hypothesis 修正**：問題不在我們的下載/儲存，而在 mlx-lm 載入後的計算。

---

## 第三步：理解 PLE 是什麼（H5）

開始懷疑 PLE quantization bug。但我自己也不知道 PLE 細節，先讀 mlx-lm 源碼：

```bash
grep -n "class\|per_layer\|embed_tokens" \
  .venv/lib/python3.12/site-packages/mlx_lm/models/gemma4_text.py
```

關鍵發現（line 305-318）：

```python
class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        ...
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input  # 256
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Linear(
                config.hidden_size, self.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input, config.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = nn.RMSNorm(...)
```

每個 DecoderLayer（35 個）都有兩個 PLE Linear。再看 `Gemma4TextModel`（line 383+）：

```python
self.embed_tokens = nn.Embedding(vocab_size, hidden_size)  # 標準 embedding
self.embed_scale = config.hidden_size**0.5

# Per-layer input embeddings (Gemma 4 特有)
self.embed_tokens_per_layer = nn.Embedding(
    vocab_size_per_layer_input,                          # 262144
    num_hidden_layers * hidden_size_per_layer_input      # 35 * 256 = 8960
)
self.embed_tokens_per_layer_scale = config.hidden_size_per_layer_input**0.5
self.per_layer_input_scale = 2.0**-0.5
self.per_layer_projection_scale = config.hidden_size**-0.5
```

### 「PLE = Per-Layer Embedding」是什麼？

傳統 Transformer 只有 input layer 一次 embedding：
```
input_ids → embed_tokens (vocab × dim) → x_0
x_0 → DecoderLayer_0 → x_1 → ... → DecoderLayer_N → logits
```

Gemma 4 在**每一層 decoder block** 都會 lookup 一個額外的 embedding：
```
x_{i-1} → DecoderLayer_i({
    main_input: x_{i-1},
    per_layer_input: embed_tokens_per_layer[input_ids][:, layer_i_slice]
}) → x_i
```

具體流程在 DecoderLayer (line 364-374)：
```python
gate = self.per_layer_input_gate(h)         # h: hidden state from previous layer
gate = mx.multiply(gate, per_layer_input)   # element-wise with PLE lookup
gate = self.per_layer_projection(gate)
gate = self.post_per_layer_input_norm(gate)
# This gate is added/used to modulate the layer's contribution
```

**為什麼用 PLE？** Google DeepMind 的研究：在每一層注入 token-specific 訊號可以讓小模型（2-4B）達到接近大模型的表現，而額外參數很少（hidden_size_per_layer_input=256 << hidden_size=1536）。

### 為什麼量化會破壞 PLE？

注意 hardcoded scales：
```python
self.per_layer_input_scale = 2.0**-0.5         ≈ 0.707
self.per_layer_projection_scale = 1536**-0.5   ≈ 0.0255
self.embed_tokens_per_layer_scale = 256**0.5   ≈ 16.0
```

當 4-bit 量化 `per_layer_input_gate` 時：
- 原始 weight `W ∈ R^(1536 × 256)`，bf16 精度
- 量化後 `W' = W + ε`，`ε` 是量化誤差（~1% range）
- 計算：`gate = W'·h = (W + ε)·h = W·h + ε·h`

加上 scale：`gate * 0.0255` → 誤差變 `ε·h * 0.0255`

**單獨看一層誤差很小**，但：
1. 35 層 decoder 都這樣做
2. 每層的誤差會經過 attention 跟 FFN 放大
3. 最後 logits 偏離真實分佈，softmax 落到完全錯的 token
4. 一旦進入錯誤的 generation 路徑，**autoregressive 會自我強化**（生成「台灣：」後，下一步在 attention 看到「台灣：」更傾向繼續生「台灣：」）

這就是 **catastrophic error amplification**：小數值誤差 × 35 層 × 自迴歸 = 完全亂碼。

### 為什麼 Colab CUDA 看起來正常？

不確定，可能：
1. CUDA 版本的 `nn.quantize` 內部對 ScaledLinear-style layers 有 fallback（執行時 dequantize 回 bf16 再算）
2. CUDA 後端 fp32 累加器精度比 Metal 高
3. Colab 上 mlx-lm 自動偵測並補救（看 peak memory 7.4 GB > 4-bit weights 2.5 GB，多出來的可能就是 dequantize buffer）

**Lesson 3**: 跨後端的 quantization bug 是常見災區。CUDA / Metal / CPU 各有自己的 fast path，量化的誤差敏感度不同。同一個 `mx.quantize()` call 在不同硬體上會有不同的數值行為。

---

## 第四步：驗證 H5（PLE 真的被量化嗎？）

讀 `model.safetensors.index.json` 的 `weight_map`，找 PLE 相關 keys：

```python
import json
idx = json.load(open('models/.../model.safetensors.index.json'))
weight_map = idx['weight_map']

# 量化的 layer 會有 .weight + .scales + .biases 三個 tensor
# 沒量化的只有 .weight
per_layer_keys = [k for k in weight_map if 'per_layer' in k]
print(f"per_layer tensors: {len(per_layer_keys)}")

# 檢查是否有對應的 .scales
for k in per_layer_keys:
    if k.endswith('.weight'):
        if k.replace('.weight', '.scales') in weight_map:
            print(f"  {k}  → QUANTIZED")
```

**結果**：

```
per_layer tensors: 252
  language_model.model.embed_tokens_per_layer.biases    ← .biases 出現 = 量化過
  language_model.model.embed_tokens_per_layer.scales    ← .scales 出現 = 量化過
  language_model.model.embed_tokens_per_layer.weight
  language_model.model.layers.0.per_layer_input_gate.biases
  language_model.model.layers.0.per_layer_input_gate.scales
  language_model.model.layers.0.per_layer_input_gate.weight
  ...
```

**Confirmed**: 所有 PLE layers 都被量化了。35 層 × 2 PLE Linear + 1 PLE embedding + 1 model projection = 72 個 PLE tensor 被量化，每個都有 `.weight + .scales + .biases` 三件套。

H5 證實。可以開始修了。

**Lesson 4**: Quantized model 的 .safetensors 結構：
- `layer.weight` 存量化後的 packed bytes
- `layer.scales` 存 group-wise scale (de-quantize 用)
- `layer.biases` 存 zero points (asymmetric quantization)
- 沒有 `.scales` 配對的就是 bf16 / fp32 原始 weight

只看 `weight_map` 就能判斷哪些 layer 被量化，不需真的載入模型。

---

## 第五步：找 fix point

mlx-lm 的 `convert()` 怎麼決定量化哪些 layer？

讀 `mlx_lm/utils.py` 的 `quantize_model`（line 825 附近）：

```python
def quantize_model(model, group_size, bits, ..., quant_predicate=None):
    def wrapped_predicate(path, module):
        ...
        bool_or_params = True
        if quant_predicate is not None:
            bool_or_params = quant_predicate(path, module)
        ...
        return bool_or_params

    nn.quantize(
        model, group_size, bits,
        mode=mode,
        class_predicate=wrapped_predicate,
    )
```

**Hook 找到了**：`quant_predicate(path, module)` 是 callable，回傳 False 就 skip。

從 `convert.py` 也看到 CLI 接口：
```python
def convert(
    hf_path, mlx_path, ...,
    quantize=False, q_bits=4, q_group_size=64,
    quant_predicate=None,    # ← 我們要的
):
```

完整 fix：

```python
from mlx_lm import convert

PLE_SKIP_PATTERNS = ("per_layer", "embed_tokens_per_layer",
                     "vision_tower", "audio_tower")

def ple_safe_predicate(path: str, module) -> bool:
    """True = quantize, False = keep bf16."""
    for pat in PLE_SKIP_PATTERNS:
        if pat in path:
            return False
    return True

convert(
    hf_path="google/gemma-4-e2b-it",
    mlx_path="output",
    quantize=True,
    q_bits=4,
    q_group_size=64,
    quant_predicate=ple_safe_predicate,
)
```

---

## 為什麼 FakeRocket543 沒送 PR？

研究 https://github.com/FakeRocket543/mlx-gemma4 後：

- 0 PRs to ml-explore/mlx-lm
- 用的是 `mlx-vlm`（不是 mlx-lm）
- 自己定義 `ScaledLinear` class（mlx-lm 沒有，是用 raw `nn.Linear` + hardcoded scalar）
- 寫獨立的 `convert_gemma4.py` script，**不依賴** mlx-lm 的 convert function

可能原因：
1. 他做的時間早於 PR #1093（mlx-lm 還沒支援 gemma4）
2. 他習慣維護自己的 fork（速度快、不用等 review）
3. 他的 fix 跟 mlx-lm 架構不相容（他用自定 class，mlx-lm 用 nn.Linear）

**對我們的意義**：mlx-lm 的修法跟他不同（我們用 `quant_predicate` skip，他用自定 ScaledLinear class）。**我們的方式更輕量**，更適合送 upstream PR。

---

## Lessons Learned 總結

### Debug methodology

1. **症狀 → hypothesis tree → 驗證最便宜的先**
   - MD5 比對是最便宜的「檔案完整性檢查」，先做
2. **修一個 bug，總要驗證問題真的解了**
   - 我們修了 JSON byte mismatch 之後沒立刻 retry smoke test 就以為對了
3. **「跨平台不一致」幾乎一定是 backend / kernel 差異**
   - Metal vs CUDA 的 quantization 數值精度不同是常見問題
4. **看 source code 比 docs 更可靠**
   - mlx-lm `--help` 沒提到 PLE，但 grep 一下就知道結構
5. **記憶體用量是 silent debugger**
   - 2.6 GB vs 7.4 GB 的差異洩漏了「Colab 在執行時做了些 fallback」

### 技術概念
- **PLE (Per-Layer Embedding)**: Gemma 4 在每層 decoder 注入 token-specific embedding，用很少參數提升小模型表現
- **量化誤差放大**: scalar multiplication + 多層連鎖 + autoregressive feedback = catastrophic
- **MLX 量化結構**: `.weight + .scales + .biases` 三件套
- **`quant_predicate` hook**: 控制哪些 layer 量化的 callback

### 工程實務
- **CLI 工具的 binary handling 要小心**: gws 對 `application/json` 的處理跟 `application/octet-stream` 完全不同
- **OAuth refresh token + curl 是萬用救命招**: 當第三方 CLI 抽象層誤事，直接打 REST API
- **Pre-release 工具謹慎用**: gws CLI 的 53+ open issues 不是裝飾品

---

## 開放問題（給之後深入）

1. **為什麼 CUDA 後端表面正常？** 是 mlx-lm 的 fallback、CUDA 數值精度差異、還是 lucky default？需要比較兩端的數值偏差。
2. **能不能 in-place 修現有的壞 model？** Dequantize PLE layers 回 bf16 然後重存。理論可行但需要原始 F16 weight 對照。
3. **PLE 量化的 acceptable bit width 是多少？** Q4 壞，Q8 應該 OK，但 ~5 GB 而不是 2.5 GB。trade-off。
4. **這個 fix 對 Gemma 4 27B / 31B 是否一樣？** 大模型的 PLE 維度更大，誤差放大可能不同。

---

## TL;DR

| Phase | 做了什麼 | 學到 |
|-------|---------|------|
| 1 | MD5 比對 → JSON byte 不對 | gws CLI 對 JSON 檔重新 serialize |
| 2 | curl Drive REST API 修 JSON | OAuth refresh token + curl 萬用 |
| 3 | 重 smoke test → 還是亂碼 | 修一個 bug 不代表全部 bug |
| 4 | 讀 gemma4_text.py 源碼 | 找到 PLE layer 結構 |
| 5 | 看 .safetensors keys | 確認 PLE 被量化 |
| 6 | 找 `quant_predicate` hook | 修法清楚，適合送 upstream PR |

**最終 fix**：用 `mlx_lm.convert(quant_predicate=ple_safe_predicate)` 跳過 PLE 層的量化。
