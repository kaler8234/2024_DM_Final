## 2024 DM Final

### 環境
* Ubuntu 22.04
* CUDA 12.2
* miniconda 虛擬環境

### 資料集準備
* [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
    下載圖片 [img_align_celeba.zip](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=drive_link&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ) 與標註檔 [identity_CelebA.txt](https://drive.google.com/file/d/1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS/view?usp=drive_link)，將圖片解壓縮後與標註檔放置於相同資料夾
* [Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)
    分別下載圖片以及標註，解壓縮後放入相同資料夾中

### 資料集處理
#### python 環境
1. 用 miniconda 開虛擬環境
```sh
conda create -n LGCN python==3.6
conda activate LGCN
```
2. 安裝所需套件
```sh
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge faiss-gpu
pip install -U scikit-learn
pip install transformers scipy matplotlib
```
3. 將 Repo 內 Data_Process 中各自對應資料集的 xx_process.py 移入前面放置資料集的資料夾中，開啟 xx_process.py 並修改開頭確認路徑 prefix 正確
4. 執行 xx_process.py 後會出現新的 xx_byclass 資料夾
5. 透過 Repo 內 Data_Process 中各自對應資料集的 xx_feature_extraction.py 移入前面放置資料集的資料夾中，執行後可以取得不同 head、tail 組合的 feature.npy 與 label.npy
6. 將本篇論文 Repo clone 下來
```sh
git clone https://github.com/espectre/GCNs_on_imbalanced_datasets.git
```
7. 使用論文 Repo 內的工具取得 knn 分群的資料
```sh
cd GCNs_on_imbalanced_datasets
python3 tools/get_knn.py \
        --data_prefix {此處填入前面資料集放置的資料夾} \
        --feats_file {此處填入 feature.npy 的路徑} \
        --topk 40
```
    這邊的輸出會產生3個檔案，需要的為 XXX_knn_index.npy 或類似字眼的檔案。

### 訓練 L-GCN
1. Clone L-GCN Github Repo
```sh
git clone https://github.com/Zhongdao/gcn_clustering.git
```
2. 執行 Train
```sh
cd gcn_clustering
python train.py \
--feat_path {feature.npy 路徑} \
--knn_graph_path {knn_index.npy 路徑} \
--labels_path {label.npy 路徑}
```

### 執行 L-GCN
1. Clone L-GCN Github Repo
```sh
git clone https://github.com/Zhongdao/gcn_clustering.git
```
2. 執行 Test
```sh
cd gcn_clustering
python test.py \
--val_feat_path {feature.npy 路徑} \
--val_knn_graph_path {knn_index.npy 路徑} \
--val_label_path {label.npy 路徑} \
--checkpoint {訓練出的模型檔}
```

> 在 L-GCN 這塊，由於原本的版本是 Python 2.7，直接使用 Python 3 的環境，部分舊版 Func 可能會報錯，下面是目前有遇過的問題。
> * has_key 相關的抱報錯，需要去報錯的地方將 code 改為 if XXX in dict 的 Python 3 語法。

### 訓練 L-GCN w/ RIWS
* 在此步驟前如還沒有 knn 的資料檔請將 feature.npy 經過 L-GCN with RIWS 內的 tools/get_knn.py 處理後取其中的 index 檔案
* 此部分 train.py 在 L-GCN with RIWS/imbalance_gcn 資料夾內

```sh
python train.py \
--feat_path {feature.npy 路徑} \
--knn_graph_path {knn_index.npy 路徑} \
--labels_path {label.npy 路徑}
```

### 執行 L-GCN w/ RIWS
* 在此步驟前如還沒有 knn 的資料檔請將 feature.npy 經過 L-GCN with RIWS 內的 tools/get_knn.py 處理後取其中的 index 檔案
* 此部分 test.py 在 L-GCN with RIWS/imbalance_gcn 資料夾內

```sh
python test.py \
--val_feat_path {feature.npy 路徑} \
--val_knn_graph_path {knn_index.npy 路徑} \
--val_labels_path {label.npy 路徑} \
--checkpoint {訓練出的模型檔}
```
