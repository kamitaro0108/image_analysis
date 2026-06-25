from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from analysis_core import detect_particles, labels_to_tiff_bytes, make_overlay, read_tiff


SAMPLE_PATH = Path("data/fluorescence_microscopy_sample.tiff")

PARAMETER_HELP = {
    "channel": "RGB画像の場合に、検出へ使うチャンネルです。蛍光が緑なら green、複数色をまとめたい場合は max が便利です。",
    "frame_index": "Zスタックや時系列TIFFなど、複数フレーム画像で解析するフレーム番号です。",
    "p_low": "コントラスト補正で黒側に丸める下位パーセンタイルです。背景をより黒くしたいときに上げます。",
    "p_high": "コントラスト補正で白側に丸める上位パーセンタイルです。明るい粒を強調したいときに下げます。",
    "background_sigma": "ゆっくり変化する背景ムラを推定して差し引く強さです。粒より十分大きい値にします。0で無効です。",
    "smooth_sigma": "しきい値処理前のノイズを抑えるぼかしです。小粒を残したい場合は小さめにします。",
    "threshold_method": "粒と背景を分けるしきい値の決め方です。otsuは自動、manualは手動、localは場所ごとに自動調整します。",
    "manual_threshold": "manual選択時のしきい値です。0に近いほど暗い粒も拾い、1に近いほど明るい粒だけを拾います。",
    "local_block_size": "local選択時に、局所しきい値を計算する範囲です。奇数で、背景ムラのスケールより小さすぎない値にします。",
    "local_offset": "local選択時の補正値です。大きくすると検出が厳しくなり、小さくすると拾いやすくなります。",
    "min_area": "これより小さい領域をノイズとして除外します。単位はピクセル数です。",
    "max_area": "これより大きい領域を除外します。凝集塊や背景の誤検出を外す用途です。",
    "clear_border": "画像の端に接している領域を除外します。切れた粒を測定から外したいときに使います。",
    "split_touching": "接触した粒を距離変換とwatershedで分離します。密集している画像では有効にします。",
    "min_distance": "粒を分離するときの中心点どうしの最小距離です。大きくすると分割されにくくなります。",
}


st.set_page_config(page_title="Fluorescence Particle Detector", layout="wide")

st.title("蛍光顕微鏡 粒検出アプリ")
st.caption("TIFFをアップロードし、輝度補正・しきい値・面積フィルタを調整しながら粒を検出します。")


@st.cache_data(show_spinner=False)
def load_uploaded_tiff(data: bytes) -> np.ndarray:
    return read_tiff(data)


@st.cache_data(show_spinner=False)
def load_sample_tiff() -> np.ndarray:
    return read_tiff(SAMPLE_PATH)


def image_shape_text(image: np.ndarray) -> str:
    return f"{image.shape}, {image.dtype}, min={image.min()}, max={image.max()}"


def build_params(image: np.ndarray) -> dict:
    st.sidebar.header("パラメータ")
    with st.sidebar.expander("パラメータの意味", expanded=False):
        for name, text in PARAMETER_HELP.items():
            st.markdown(f"**{name}**: {text}")

    channels = ["green", "red", "blue", "max", "mean"]
    if image.ndim == 2:
        channels = ["green"]

    frame_count = image.shape[0] if image.ndim in (3, 4) and image.shape[-1] not in (3, 4) else 1

    params = {
        "channel": st.sidebar.selectbox("検出チャンネル", channels, index=0, help=PARAMETER_HELP["channel"]),
        "frame_index": 0,
        "p_low": st.sidebar.slider("黒側補正 p_low", 0.0, 20.0, 1.0, 0.5, help=PARAMETER_HELP["p_low"]),
        "p_high": st.sidebar.slider("白側補正 p_high", 80.0, 100.0, 99.8, 0.1, help=PARAMETER_HELP["p_high"]),
        "background_sigma": st.sidebar.slider("背景差し引き background_sigma", 0.0, 100.0, 20.0, 1.0, help=PARAMETER_HELP["background_sigma"]),
        "smooth_sigma": st.sidebar.slider("平滑化 smooth_sigma", 0.0, 8.0, 1.2, 0.1, help=PARAMETER_HELP["smooth_sigma"]),
        "threshold_method": st.sidebar.selectbox("しきい値方式", ["otsu", "manual", "local"], help=PARAMETER_HELP["threshold_method"]),
        "manual_threshold": st.sidebar.slider("手動しきい値", 0.0, 1.0, 0.25, 0.01, help=PARAMETER_HELP["manual_threshold"]),
        "local_block_size": st.sidebar.slider("局所範囲 block size", 11, 301, 51, 2, help=PARAMETER_HELP["local_block_size"]),
        "local_offset": st.sidebar.slider("局所補正 offset", -0.2, 0.2, 0.0, 0.005, help=PARAMETER_HELP["local_offset"]),
        "min_area": st.sidebar.slider("最小面積 min_area", 1, 2000, 20, 1, help=PARAMETER_HELP["min_area"]),
        "max_area": st.sidebar.slider("最大面積 max_area", 10, 50000, 2000, 10, help=PARAMETER_HELP["max_area"]),
        "clear_border": st.sidebar.checkbox("端の粒を除外", value=False, help=PARAMETER_HELP["clear_border"]),
        "split_touching": st.sidebar.checkbox("接触粒を分離", value=True, help=PARAMETER_HELP["split_touching"]),
        "min_distance": st.sidebar.slider("分離距離 min_distance", 1, 80, 5, 1, help=PARAMETER_HELP["min_distance"]),
    }

    if frame_count > 1:
        params["frame_index"] = st.sidebar.slider("フレーム番号", 0, frame_count - 1, 0, 1, help=PARAMETER_HELP["frame_index"])

    return params


uploaded = st.file_uploader("TIFFファイルをアップロード", type=["tif", "tiff"])
use_sample = st.checkbox("サンプル画像を使う", value=uploaded is None)

if uploaded is not None:
    image = load_uploaded_tiff(uploaded.getvalue())
    source_name = uploaded.name
elif use_sample and SAMPLE_PATH.exists():
    image = load_sample_tiff()
    source_name = str(SAMPLE_PATH)
else:
    st.info("解析するTIFFファイルをアップロードしてください。")
    st.stop()

params = build_params(image)

st.subheader("入力画像")
st.write(f"`{source_name}`")
st.code(image_shape_text(image), language="text")

with st.spinner("粒を検出しています..."):
    result = detect_particles(image, **params)
    overlay = make_overlay(result.corrected, result.labels)

threshold_text = "local" if result.threshold_value is None else f"{result.threshold_value:.4f}"

metric_cols = st.columns(4)
metric_cols[0].metric("検出数", f"{len(result.measurements):,}")
metric_cols[1].metric("しきい値", threshold_text)
metric_cols[2].metric("平均面積", "0" if result.measurements.empty else f"{result.measurements['area'].mean():.1f} px")
metric_cols[3].metric("平均輝度", "0" if result.measurements.empty else f"{result.measurements['mean_intensity'].mean():.3f}")

tabs = st.tabs(["検出結果", "前処理", "マスク", "測定表", "ダウンロード"])

with tabs[0]:
    left, right = st.columns(2)
    left.image(result.corrected, caption="背景補正後", clamp=True, use_container_width=True)
    right.image(overlay, caption="検出輪郭", use_container_width=True)

with tabs[1]:
    cols = st.columns(3)
    cols[0].image(result.selected, caption="選択チャンネル", clamp=True, use_container_width=True)
    cols[1].image(result.normalized, caption="コントラスト補正後", clamp=True, use_container_width=True)
    cols[2].image(result.smoothed, caption="平滑化後", clamp=True, use_container_width=True)

with tabs[2]:
    cols = st.columns(3)
    cols[0].image(result.mask.astype(float), caption="初期マスク", clamp=True, use_container_width=True)
    cols[1].image(result.cleaned.astype(float), caption="クリーニング後", clamp=True, use_container_width=True)
    cols[2].image(result.labels, caption="ラベル画像", clamp=True, use_container_width=True)

with tabs[3]:
    st.dataframe(result.measurements, use_container_width=True, hide_index=True)
    if not result.measurements.empty:
        st.bar_chart(result.measurements["area"], x_label="particle", y_label="area px")

with tabs[4]:
    csv_data = result.measurements.to_csv(index=False).encode("utf-8-sig")
    label_data = labels_to_tiff_bytes(result.labels)
    st.download_button("測定CSVをダウンロード", csv_data, file_name="particle_measurements.csv", mime="text/csv")
    st.download_button("ラベルTIFFをダウンロード", label_data, file_name="particle_labels.tiff", mime="image/tiff")

