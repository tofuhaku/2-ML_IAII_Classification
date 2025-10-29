## 概述
此專案旨在建立一個深度學習模型，用於對辛普森家庭中的 50 位角色進行圖像分類。整個工作流程涵蓋資料前處理、模型建構、模型訓練，以及最終的預測生成。專案採用 PyTorch 框架，並利用預訓練模型進行遷移學習，以達到更高的分類準確率。以下將詳細分析各個模組的功能與實現方式。

---

## 1. 核心設定 (`config.py`)

`config.py` 檔案是整個專案的中央設定檔，定義所有重要的超參數和路徑，使得程式碼更易於管理和修改。

- **硬體與路徑設定**:
    - `DEVICE`: 自動偵測是否有可用的 CUDA GPU，若有則使用 GPU 進行運算，否則使用 CPU。
    - `TRAIN_PATH`, `TEST_PATH`, `BACKGROUND_DIR`: 分別定義訓練資料集、測試資料集以及用於資料增強的背景圖片的路徑。
    - `MODEL_WEIGHTS_PATH`, `OUTPUT_CSV_PATH`: 指定訓練後最佳模型的儲存路徑 (`best_model.pth`) 和最終預測結果的輸出路徑 (`submission.csv`)。

- **模型與訓練超參數**:
    - `BATCH_SIZE`: 設定每個 batch 的訓練樣本數為 128。
    - `NUM_EPOCHS`: 訓練的總 epoch 設定為 50。
    - `NUM_WORKERS`: 資料載入時使用的子行程數為 8，以加速資料處理。
    - `NUM_CLASSES`: 分類的目標類別數，設定為 50，共有 50 個角色。
    - `LEARNING_RATE`: 學習率設定為 `1e-4`。
    - `IMAGE_SIZE`: 所有輸入圖片將被統一調整為 `224x224` 像素。
    - `VAL_SPLIT`: 從訓練資料集中切分 20% 作為驗證集。

- **類別對應**:
    - `CHARACTER_NAMES`: 包含 50 個角色名稱的列表。
    - `IDX_TO_CLASS` 和 `CLASS_TO_IDX`: 提供索引和類別名稱之間的雙向映射，方便在訓練和預測時進行轉換。

---

## 2. 資料前處理與載入

資料的準備工作由 `calculate_stats.py` 和 `data_loader.py` 兩個檔案共同完成。

### 2.1. 資料集統計計算 (`calculate_stats.py`)

此腳本的目的是計算訓練資料集的像素平均值 (mean) 和標準差 (std)，用於後續資料的的 `Normalize` 轉換。

- **主要功能**:
    1.  讀取 `config.TRAIN_PATH` 路徑下的所有訓練圖片。
    2.  將圖片大小統一調整並轉換為 `Tensor`。
    3.  遍歷整個資料集，逐批次計算所有像素在 RGB 三個通道上的總和與平方和。
    4.  根據累加的數值，最終計算出整個資料集的平均值和標準差。

這個標準化步驟能加速模型訓練並更穩定地收斂。

### 2.2. 資料載入與增強 (`data_loader.py`)

此檔案定義資料集的結構、資料增強的策略以及資料載入器 (DataLoader) 的生成。

- **`SimpsonsDataset` 類別**:
    - 繼承自 PyTorch 的 `Dataset` 類別，用於讀取圖片和對應的標籤。
    - 針對訓練集 (`is_train=True`)，它會遍歷各角色資料夾，讀取圖片路徑並將角色名稱轉換為對應的索引標籤。
    - 針對測試集 (`is_train=False`)，它會直接讀取圖片，並將檔名（不含副檔名）作為圖片 ID。

- **資料增強 (Data Augmentation)**:
    - 透過 `get_transforms` 函式定義資料增強流程，僅應用於訓練集，以增加資料多樣性、提高模型的泛化能力。主要包含：
        - `AddRandomBackground`: 以 70% 的機率隨機替換圖片背景。
        - 幾何變換：隨機水平翻轉、垂直翻轉、旋轉。
        - 顏色變換：隨機調整亮度、對比度、飽和度和色調，以及隨機灰階化。
        - 噪聲注入：實現多種自定義的噪聲注入類別，如高斯噪聲 (`AddGaussianNoise`)、斑點噪聲 (`AddSpeckleNoise`) 等，以模擬真實世界的圖像干擾。
        - 標準化：使用 `calculate_stats.py` 計算出的平均值和標準差對圖像進行標準化。
    - 驗證集和測試集僅進行必要的尺寸調整和標準化，以確保評估的一致性。

- **`get_dataloaders` 和 `get_test_loader` 函式**:
    - `get_dataloaders`: 負責創建訓練和驗證資料載入器。它會先將完整資料集依 `VAL_SPLIT` 比例分割為訓練子集和驗證子集，並分別應用訓練轉換和驗證轉換。
    - `get_test_loader`: 負責創建測試資料載入器。

---

## 3. 模型架構 (`model.py`)

此檔案定義用於分類任務的卷積神經網絡 (CNN) 模型架構。專案採用遷移學習的方法，並可以使用兩種常見的預訓練模型：`ResNet` 和 `EfficientNet`。

- **`SimpsonsClassifier`**:
    - 基於 `ResNet-50` 模型，並載入在 ImageNet 上預訓練的權重。
    - **遷移學習**:
        - **凍結層**: 為了保留預訓練模型學習到的底層特徵（如邊緣、紋理），模型凍結大部分早期層的參數 (`param.requires_grad = False`)，僅讓最後 20 層的參數在訓練中進行微調。
        - **修改 classification head**: 原始 `ResNet-50` 的最後一層 (fc layer) 被替換為一個新的 classification head 。這個 classification head 包含 Dropout 層以防止過擬合，一個 ReLU 激活函式，以及最終輸出為 `NUM_CLASSES` (50) 個類別的全連接層。

- **`EfficientNetClassifier`**:
    - 基於 `EfficientNet-B0` 模型，同樣載入 pretrained weight。
    - **架構修改**: 與 `SimpsonsClassifier` 類似，它也替換原始模型的分類器部分 (`self.backbone.classifier`)，換成與前者結構相同的自定義 classification head。
    - **備援機制**: 程式碼中包含一個 `try-except` 區塊，若 `EfficientNet` 模型因某些原因無法載入，它會自動切換回使用 `ResNet-34` 作為備用模型，增加程式碼的穩健性。

---

## 4. 模型訓練 (`train.py`)

此腳本是整個專案的核心，負責執行模型的訓練、驗證、儲存和結果可視化。

- **`train_model` 函式**:
    - **損失函式與優化器**: 使用 `CrossEntropyLoss` 作為損失函式，這適用於多分類任務。優化器選用 `AdamW`，這是在 `Adam` 基礎上改進的版本，能更好地處理權重衰減。
    - **學習率排程器**: 使用 `CosineAnnealingLR`，它能讓學習率在訓練過程中週期性地下降，有助於模型跳出局部最優點並收斂得更好。
    - **訓練與驗證迴圈**:
        - 在每個 epoch 中，模型首先進入訓練模式 (`model.train()`)，遍歷訓練資料載入器，計算損失、反向傳播並更新權重。
        - 隨後，模型進入評估模式 (`model.eval()`)，遍歷驗證資料載入器，計算驗證集上的損失和準確率，此階段不進行梯度更新。
        - 使用 `tqdm` 套件顯示進度條，並即時更新當前的損失和準確率，方便監控訓練過程。
    - **最佳模型儲存**: 在每個 epoch 結束後，如果當前的驗證準確率 (`val_acc`) 超過歷史最佳值，則將模型的狀態 (`model.state_dict()`) 儲存到 `config.MODEL_WEIGHTS_PATH` 指定的檔案中。

- **`plot_curves` 函式**:
    - 訓練結束後，此函式會利用 `matplotlib` 將訓練過程中的損失和準確率歷史記錄繪製成圖表，並儲存為 `training_curves.png`。這有助於分析模型的學習狀況，例如是否出現過擬合或欠擬合。

- **`main` 函式**:
    - 串連整個訓練流程：
        1.  檢查或計算資料集的統計數據。
        2.  初始化資料載入器。
        3.  初始化 `EfficientNetClassifier` 模型並移至指定設備 (GPU/CPU)。
        4.  呼叫 `train_model` 執行訓練。
        5.  呼叫 `plot_curves` 繪製結果圖。

---

## 5. 預測與提交 (`predict.py`)

當模型訓練完成後，此腳本用於對測試集進行預測，並生成符合提交格式的 CSV 檔案。

- **`generate_predictions` 函式**:
    - 將模型設為評估模式 (`model.eval()`)。
    - 在 `torch.no_grad()` 上下文中執行，以關閉梯度計算，節省記憶體並加速預測。
    - 遍歷測試資料載入器，將模型的輸出通過 `torch.max` 找到最高機率的類別索引。
    - 使用 `config.IDX_TO_CLASS` 將索引轉換回對應的角色名稱。
    - 收集所有圖片的 ID 和預測的角色名稱。

- **`main` 函式**:
    - **前置檢查**: 首先檢查必要的模型權重檔案和統計數據檔案是否存在，若缺少則終止程式並提示使用者先執行訓練。
    - **流程**:
        1.  載入資料集統計數據。
        2.  初始化與訓練時相同架構的模型 (`EfficientNetClassifier`)。
        3.  載入已訓練好的最佳模型權重 (`best_model.pth`)。
        4.  取得測試資料載入器。
        5.  呼叫 `generate_predictions` 產生預測結果。
        6.  將結果整理成 `pandas.DataFrame`，並根據圖片 ID 進行排序，最終儲存為 `submission.csv` 檔案。

***
## 結論

本次 lab 嘗試使用 `EfficientNetClassifier` 和 `ResNet-50`，在本地的 RTX3060 進行訓練。嘗試幾次發現由於資料擴充複雜，`ResNet-50` 訓練的時間超乎預期的長（約 50min / epoch），在相同的資料擴充下效果也沒有特別突出，因此最後使用 `EfficientNetClassifier`，耗時較短（約 20min / epoch）效果也算優秀，兩者模型使用的參數量相差約一倍。本次實驗的瓶頸其實是硬體限制，在有限的資源下要取捨 performance 和時間成本的消耗，最後是訓練約 15 小時達到 test dataset 辨識正確率 93%。

<img src=".\training_curves.png" alt="training_curves" style="zoom:60%;" />