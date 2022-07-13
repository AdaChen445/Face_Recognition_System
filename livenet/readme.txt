活體辨識模型大禮包
===
# 操作說明:
## 影片取樣:
step1. 將待取樣影片放入input_video資料夾中
step2. anaconda prompt開啟環境，輸入python gather_data.py
step3. 取樣好的圖片會儲存在output_face中，使用Renamer批次重新命名，以免彙整資料時名稱相同覆蓋掉
step4. 將重新命名好的圖片移到dataset對應的資料夾中，成為訓練模型樣本

## 訓練模型:
step1. anaconda prompt開啟環境，輸入python train.py
step2. 訓練時會出現一堆數值，但只要注意val_accuracy驗證準確度就好，理論上越接近1越好
step3. 訓練結束後會出現模型數據，理論上fake/real/precision/recall矩陣中的四項數值越接近1越好

## 模型測試:
step1. anaconda prompt開啟環境，輸入python demo.py
step2. 在各方向光源、環境、手機顯示照片、證件實體照片等測試模型效果
===============================================================
# 其他提醒:
## 關於取樣:
1. gather_data.py中，可以調整framejump參數改變每幾幀取一張人臉，預設是每10張
2. 若取樣的人臉會向左或右轉向90度，可以註解掉gather_data.py中第30-32列程式碼
3. 目前缺:背光源(臉黑)、手機顯示照片時螢幕亮度調高、證件或其他實體照這幾種狀況的樣本

## 關於模型訓練:
1. train.py中兩大重要參數，BS:每次訓練的分群、EPOCHS:訓練次數；BS對訓練效果目前不清楚，但為總樣本數的因數比較好；EPOCHS越大準確度會越高，但有可能導致overfitting
2. val_accuracy過高時會出現overfitting，意思是模型只認得訓練樣本的形狀而使辨識失真，總之這些數值並非越高越好，需要持續trail and error
3. 目前測試辨識失效狀況:背光源(臉黑)、手機顯示照片時螢幕亮度調高、證件或其他實體照
4. 若測試時還有其他失效狀況，建議將當下狀況取為樣本

===============================================================

# 文件說明:
livenet
├─__pycache__:直譯快取文件，不用理會
├─dataset:訓練模型時讀取之樣本，裡面資料夾的名字就是辨識時的標示名稱
│  ├─real:真實人臉樣本
│  └─fake:翻攝人臉樣本
├─detector:人臉偵測預訓練模型，不用理會
├─input_video:存放取樣用影片，執行gather_data時會從這抓影片取樣
├─other_dataset:上次meeting取樣好的照片，三個人的都在裡面
├─output_face:存放取樣好的照片，執行gather_data時會存照片於此
├─demo.py:使用筆電攝影機示範及測試模型實際使用之程式
├─gatherData.py:用以從樣本影片中抓取人臉之程式
├─le.pickle:訓練時自動產出相依檔案，不用理會
├─liveness.model:訓練好自動產出之模型，不用理會
├─net.py:keras撰寫之活體辨識模型架構
├─plot.png:模型訓練過程曲線圖
├─readme.txt:說明文件
├─ReNamer.exe:用以將圖片批次命名的好用小工具
└─train.py:訓練模型用程式