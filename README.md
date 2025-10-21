# The Simpsons Character Recognition System

一個使用CNN深度學習技術來辨認50個The Simpsons影集角色的完整影像辨認系統。

## 🎯 系統特色

- **50個角色辨認**: 支援完整的50個The Simpsons主要角色
- **先進的CNN架構**: 使用ResNet50作為backbone，結合pre-training和fine-tuning
- **資料擴充**: 包含翻轉、顏色調整、雜訊、透視變換等多種資料擴充技術
- **模組化設計**: 清晰的代碼結構，易於維護和擴展
- **完整的訓練流程**: 包含early stopping、learning rate scheduling、class weight balancing
- **即時預測**: 支援單張圖片預測和批次推論

## 📁 檔案結構

```
├── main.py              # 主執行檔案
├── config.py           # 配置參數
├── model.py            # CNN模型架構
├── data_preprocess.py  # 資料預處理和擴充
├── train.py            # 訓練腳本
├── inference.py        # 推論和CSV生成
├── requirements.txt    # 依賴套件
└── README.md          # 說明文件
```

## 🔧 環境配置

### 1. 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 2. 資料集結構

確保你的資料集結構如下：

```
dataset/
├── train/
│   └── train/
│       ├── abraham_grampa_simpson/
│       ├── apu_nahasapeemapetilon/
│       ├── bart_simpson/
│       └── ... (50個角色資料夾)
└── test-renamed_images/
    ├── 1.jpg
    ├── 2.jpg
    └── ... (測試圖片)
```

## 🚀 使用方法

### 訓練模型

```bash
# 完整訓練和測試流程
python main.py --mode both

# 僅訓練
python main.py --mode train

# 僅測試（需要已訓練的模型）
python main.py --mode test
```

### 預測單張圖片

```bash
python main.py --mode predict --image path/to/your/image.jpg
```

### 資料分析

```bash
# 查看系統資訊和支援的角色列表
python main.py --mode info

# 分析資料集
python main.py --mode analyze
```

## 🏗️ 模型架構

### 主要特色：

1. **Pre-trained ResNet50**: 使用ImageNet預訓練權重
2. **兩階段訓練**:
   - 階段1: 凍結backbone，僅訓練分類器
   - 階段2: 解凍backbone進行fine-tuning
3. **進階資料擴充**:
   - 隨機旋轉、翻轉、色彩調整
   - 透視變換、仿射變換
   - 高斯雜訊添加
4. **損失函數**: 支援Cross Entropy、Focal Loss、Label Smoothing

### 超參數配置：

- 圖片尺寸: 224×224
- Batch Size: 32
- 學習率: 0.001 (AdamW優化器)
- 訓練輪數: 100 (含early stopping)
- Dropout: 0.5

## 📊 訓練特色

- **Early Stopping**: 防止過擬合
- **Learning Rate Scheduling**: 自動調整學習率
- **Class Weight Balancing**: 處理類別不平衡問題
- **Gradient Clipping**: 防止梯度爆炸
- **模型檢查點**: 自動保存最佳模型

## 📈 輸出結果

系統會生成以下檔案：

1. **submission.csv**: 包含id和character兩欄的預測結果
2. **training_history.png**: 訓練過程的loss和accuracy圖表
3. **best_model.pth**: 最佳模型權重
4. **checkpoints/**: 訓練檢查點

## 🎭 支援的角色 (50個)

1. abraham_grampa_simpson
2. agnes_skinner
3. apu_nahasapeemapetilon
4. barney_gumble
5. bart_simpson
6. brandine_spuckler
7. carl_carlson
8. charles_montgomery_burns
9. chief_wiggum
10. cletus_spuckler
11. comic_book_guy
12. disco_stu
13. dolph_starbeam
14. duff_man
15. edna_krabappel
16. fat_tony
17. gary_chalmers
18. gil
19. groundskeeper_willie
20. homer_simpson
21. jimbo_jones
22. kearney_zzyzwicz
23. kent_brockman
24. krusty_the_clown
25. lenny_leonard
26. lionel_hutz
27. lisa_simpson
28. lunchlady_doris
29. maggie_simpson
30. marge_simpson
31. martin_prince
32. mayor_quimby
33. milhouse_van_houten
34. miss_hoover
35. moe_szyslak
36. ned_flanders
37. nelson_muntz
38. otto_mann
39. patty_bouvier
40. principal_skinner
41. professor_john_frink
42. rainier_wolfcastle
43. ralph_wiggum
44. selma_bouvier
45. sideshow_bob
46. sideshow_mel
47. snake_jailbird
48. timothy_lovejoy
49. troy_mcclure
50. waylon_smithers

## ⚙️ 技術細節

### 資料擴充策略

為了應對測試集中的變形、顏色變化、雜訊干擾等挑戰，系統實作了全面的資料擴充：

1. **幾何變換**:
   - 隨機旋轉 (±15度)
   - 水平翻轉 (50%機率)
   - 隨機透視變換
   - 仿射變換 (平移、縮放)

2. **顏色調整**:
   - 亮度調整 (±20%)
   - 對比度調整 (±20%)
   - 飽和度調整 (±20%)
   - 色調調整 (±10%)

3. **雜訊處理**:
   - 高斯雜訊添加
   - Dropout2D防止過擬合

### Pre-training策略

1. **冷啟動**: 使用ImageNet預訓練的ResNet50
2. **漸進式解凍**: 先訓練分類器，再fine-tune整個網路
3. **學習率調節**: Fine-tuning時使用較小的學習率

## 🛠️ 自定義配置

可以透過修改`config.py`來調整超參數：

- 調整圖片尺寸: `IMG_SIZE`
- 修改batch size: `BATCH_SIZE`
- 設定訓練輪數: `EPOCHS`
- 調整學習率: `LEARNING_RATE`
- 配置資料擴充強度: `ROTATION_DEGREES`, `COLOR_JITTER_*`

## 🔍 故障排除

### 常見問題：

1. **記憶體不足**: 減少`BATCH_SIZE`
2. **訓練太慢**: 確認CUDA是否可用，或減少`IMG_SIZE`
3. **過擬合**: 增加dropout率或資料擴充強度
4. **欠擬合**: 減少dropout率或增加模型複雜度

### 硬體需求：

- **最低配置**: 8GB RAM, CPU訓練
- **建議配置**: 16GB RAM, NVIDIA GPU (8GB+ VRAM)
- **儲存空間**: 至少5GB (含資料集和模型)

## 📄 License

MIT License

## 🤝 貢獻

歡迎提交Pull Request或Issue來改善這個項目！