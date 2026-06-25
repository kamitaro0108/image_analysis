# 蛍光顕微鏡 粒検出Webアプリ

TIFF画像をアップロードし、輝度補正・しきい値・面積フィルタなどを調整しながら粒を検出するStreamlitアプリです。

## 1. 前提

- Python 3.10 以上を想定しています。
- WindowsのCommand Promptを想定しています。
- まずCommand Promptを開き、このプロジェクトを置いたフォルダへ移動してください。

```bat
cd /d "プロジェクトを置いたフォルダのパス"
```

例:

```bat
cd /d "C:\path\to\image_analysis"
```

## 2. 仮想環境を作成

```bat
python -m venv .venv
```

## 3. 仮想環境を有効化

```bat
.venv\Scripts\activate.bat
```

有効化できると、行の先頭に `(.venv)` のような表示が付きます。

## 4. 必要パッケージをインストール

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. アプリを起動

一番簡単な起動方法:

```bat
start_webapp.bat
```

または、Streamlitを直接起動する場合:

```bat
streamlit run app.py
```

仮想環境を有効化していない状態で直接起動したい場合:

```bat
.venv\Scripts\streamlit.exe run app.py
```

## 6. ブラウザで開く

起動すると、ターミナルに次のようなURLが表示されます。

```text
http://localhost:8501
```

ブラウザでこのURLを開きます。

## 7. 使い方

1. `TIFFファイルをアップロード` から解析したいTIFF画像を選びます。
2. サイドバーの `まずはこの順番で調整` を見ながらパラメータを調整します。
3. `検出結果` タブで、黄色い輪郭が粒に合っているか確認します。
4. `前処理` タブで、背景補正やノイズ低減の効き方を確認します。
5. `マスク` タブで、粒候補が白く、背景が黒くなっているか確認します。
6. `測定表` タブで、検出された粒の面積・中心位置・輝度を確認します。
7. `ダウンロード` タブからCSVやラベルTIFFを保存します。

## 8. 終了方法

アプリを起動しているターミナルで `Ctrl + C` を押します。

## 9. 次回以降の起動

すでに `.venv` とパッケージが入っている場合は、次回からはこれだけで起動できます。

```bat
start_webapp.bat
```
