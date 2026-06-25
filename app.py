from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from analysis_core import detect_particles, labels_to_tiff_bytes, make_overlay, read_tiff


SAMPLE_PATH = Path("data/fluorescence_microscopy_sample.tiff")

PARAMETER_HELP = {
    "channel": "どの色の明るさを使って粒を探すかを決めます。緑の蛍光なら green、複数色をまとめたいときは max が便利です。",
    "frame_index": "Zスタックや時系列TIFFで、何枚目の画像を解析するかを選びます。",
    "p_low": "暗い側の明るさ補正です。上げると背景がより黒くなり、弱い背景ノイズを抑えやすくなります。",
    "p_high": "明るい側の明るさ補正です。下げると明るい粒が強調されますが、強すぎると粒の中が白飛びします。",
    "background_sigma": "ゆっくり変化する背景ムラを引く強さです。値を大きくすると広い背景ムラを取り除きます。0で無効です。",
    "smooth_sigma": "しきい値処理の前に画像を少しぼかしてノイズを減らします。上げすぎると小さい粒が消えます。",
    "threshold_method": "背景と粒を分ける方法です。otsuは自動、manualは自分で調整、localは場所ごとにしきい値を変えます。",
    "manual_threshold": "manual のときだけ効く明るさの境界です。下げると検出が増え、上げると明るい粒だけ残ります。",
    "local_block_size": "local のときだけ効く、周囲何ピクセルを見てしきい値を決めるかです。奇数で指定します。",
    "local_offset": "local のときだけ効く微調整です。上げると厳しくなり、下げると拾いやすくなります。",
    "min_area": "これより小さい領域をノイズとして捨てます。小さい点ノイズが多いときに上げます。",
    "max_area": "これより大きい領域を捨てます。大きな背景領域や凝集塊を外したいときに下げます。",
    "clear_border": "画像の端に接している領域を除外します。端で切れている粒を数えたくないときに使います。",
    "split_touching": "くっついた粒を分ける処理です。密集して1つに見えてしまう粒が多いときに有効です。",
    "min_distance": "粒を分けるとき、中心どうしをどれくらい離れているとみなすかです。上げると分割されにくくなります。",
}

WORKFLOW_STEPS = [
    ("1. チャンネルを選ぶ", "粒が一番明るく見える色を選びます。迷ったら green または max から試します。"),
    ("2. 前処理を整える", "p_low / p_high / background_sigma / smooth_sigma で、背景を暗く、粒を見やすくします。"),
    ("3. しきい値を決める", "mask に粒だけが白く出るように threshold_method と manual_threshold などを調整します。"),
    ("4. ノイズを消す", "min_area と max_area で小さすぎる点や大きすぎる誤検出を除きます。"),
    ("5. くっついた粒を分ける", "split_touching と min_distance で、密集した粒の分割具合を調整します。"),
]


st.set_page_config(page_title="Fluorescence Particle Detector", layout="wide")


@st.cache_data(show_spinner=False)
def load_uploaded_tiff(data: bytes) -> np.ndarray:
    return read_tiff(data)


@st.cache_data(show_spinner=False)
def load_sample_tiff() -> np.ndarray:
    return read_tiff(SAMPLE_PATH)


def image_shape_text(image: np.ndarray) -> str:
    return f"{image.shape}, {image.dtype}, min={image.min()}, max={image.max()}"


def explain_box(title: str, body: str, params: list[str], tips: list[str] | None = None) -> None:
    st.markdown(f"#### {title}")
    st.write(body)
    st.markdown("**主に効くパラメータ**")
    st.write(" / ".join(f"`{param}`" for param in params))
    if tips:
        st.markdown("**調整の目安**")
        for tip in tips:
            st.markdown(f"- {tip}")


def build_params(image: np.ndarray) -> dict:
    st.sidebar.header("パラメータ")

    with st.sidebar.expander("まずはこの順番で調整", expanded=True):
        for title, text in WORKFLOW_STEPS:
            st.markdown(f"**{title}**")
            st.caption(text)

    with st.sidebar.expander("各パラメータの意味", expanded=False):
        for name, text in PARAMETER_HELP.items():
            st.markdown(f"**{name}**")
            st.caption(text)

    channels = ["green", "red", "blue", "max", "mean"]
    if image.ndim == 2:
        channels = ["green"]

    frame_count = image.shape[0] if image.ndim in (3, 4) and image.shape[-1] not in (3, 4) else 1

    params = {
        "channel": st.sidebar.selectbox("検出チャンネル", channels, index=0, help=PARAMETER_HELP["channel"]),
        "frame_index": 0,
        "p_low": st.sidebar.slider("暗い側の補正 p_low", 0.0, 20.0, 1.0, 0.5, help=PARAMETER_HELP["p_low"]),
        "p_high": st.sidebar.slider("明るい側の補正 p_high", 80.0, 100.0, 99.8, 0.1, help=PARAMETER_HELP["p_high"]),
        "background_sigma": st.sidebar.slider("背景ムラ除去 background_sigma", 0.0, 100.0, 20.0, 1.0, help=PARAMETER_HELP["background_sigma"]),
        "smooth_sigma": st.sidebar.slider("ノイズ低減 smooth_sigma", 0.0, 8.0, 1.2, 0.1, help=PARAMETER_HELP["smooth_sigma"]),
        "threshold_method": st.sidebar.selectbox("しきい値の決め方", ["otsu", "manual", "local"], help=PARAMETER_HELP["threshold_method"]),
        "manual_threshold": st.sidebar.slider("手動しきい値", 0.0, 1.0, 0.25, 0.01, help=PARAMETER_HELP["manual_threshold"]),
        "local_block_size": st.sidebar.slider("局所しきい値の範囲", 11, 301, 51, 2, help=PARAMETER_HELP["local_block_size"]),
        "local_offset": st.sidebar.slider("局所しきい値の補正", -0.2, 0.2, 0.0, 0.005, help=PARAMETER_HELP["local_offset"]),
        "min_area": st.sidebar.slider("最小面積 min_area", 1, 2000, 20, 1, help=PARAMETER_HELP["min_area"]),
        "max_area": st.sidebar.slider("最大面積 max_area", 10, 50000, 2000, 10, help=PARAMETER_HELP["max_area"]),
        "clear_border": st.sidebar.checkbox("端に接する粒を除外", value=False, help=PARAMETER_HELP["clear_border"]),
        "split_touching": st.sidebar.checkbox("接触した粒を分離", value=True, help=PARAMETER_HELP["split_touching"]),
        "min_distance": st.sidebar.slider("分離距離 min_distance", 1, 80, 5, 1, help=PARAMETER_HELP["min_distance"]),
    }

    if frame_count > 1:
        params["frame_index"] = st.sidebar.slider("フレーム番号", 0, frame_count - 1, 0, 1, help=PARAMETER_HELP["frame_index"])

    return params


st.title("蛍光顕微鏡 粒検出アプリ")
st.caption("TIFFをアップロードし、画像を見ながら粒検出のパラメータを調整できます。")

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
    explain_box(
        "検出結果の見方",
        "左は検出に使った補正済み画像、右は検出された粒の輪郭を黄色で重ねた画像です。最終的に数えられる粒は、この黄色い輪郭の領域です。",
        ["threshold_method", "manual_threshold", "min_area", "max_area", "split_touching", "min_distance"],
        [
            "粒が足りない場合は、しきい値を下げる、または min_area を小さくします。",
            "背景やノイズを拾いすぎる場合は、しきい値を上げる、または min_area を大きくします。",
            "くっついた粒が1つに数えられる場合は split_touching を有効にし、min_distance を小さめにします。",
        ],
    )
    left, right = st.columns(2)
    left.image(result.corrected, caption="背景補正後: 検出に使う明るさ画像", clamp=True, use_container_width=True)
    right.image(overlay, caption="検出輪郭: 黄色が最終的に検出された粒", use_container_width=True)

with tabs[1]:
    explain_box(
        "前処理で行っていること",
        "前処理は、しきい値で粒と背景を分けやすくする準備です。ここが整うと、マスク画像で粒だけが白くなりやすくなります。",
        ["channel", "p_low", "p_high", "background_sigma", "smooth_sigma"],
        [
            "選択チャンネルで粒が見えにくい場合は channel を変えます。",
            "背景が明るく残る場合は p_low や background_sigma を上げます。",
            "細かい点ノイズが多い場合は smooth_sigma を少し上げます。",
            "小さい粒まで見たい場合は smooth_sigma を上げすぎないようにします。",
        ],
    )
    cols = st.columns(3)
    cols[0].image(result.selected, caption="1. 選択チャンネル: 解析に使う元の明るさ", clamp=True, use_container_width=True)
    cols[1].image(result.normalized, caption="2. コントラスト補正後: 粒を見やすくした画像", clamp=True, use_container_width=True)
    cols[2].image(result.smoothed, caption="3. 平滑化後: ノイズを少し減らした画像", clamp=True, use_container_width=True)

with tabs[2]:
    explain_box(
        "マスク画像で行っていること",
        "マスクは、粒候補を白、背景を黒で表した画像です。検出の失敗原因はこのタブを見ると分かりやすいです。",
        ["threshold_method", "manual_threshold", "local_block_size", "local_offset", "min_area", "max_area", "clear_border"],
        [
            "初期マスクで粒が黒いままなら、しきい値が高すぎます。",
            "初期マスクで背景まで白いなら、しきい値が低すぎます。",
            "クリーニング後に小さな点が消えるのは min_area の効果です。",
            "ラベル画像では、つながっている粒候補に番号が付きます。色の塊1つが1粒として数えられます。",
        ],
    )
    cols = st.columns(3)
    cols[0].image(result.mask.astype(float), caption="初期マスク: しきい値で白黒に分けた直後", clamp=True, use_container_width=True)
    cols[1].image(result.cleaned.astype(float), caption="クリーニング後: 小さいノイズや穴を整理した後", clamp=True, use_container_width=True)
    cols[2].image(result.labels, caption="ラベル画像: 粒ごとに番号を付けた最終候補", clamp=True, use_container_width=True)

with tabs[3]:
    explain_box(
        "測定表の見方",
        "1行が1つの粒です。面積、中心位置、平均輝度、最大輝度、円相当径を確認できます。",
        ["min_area", "max_area", "split_touching", "min_distance"],
        [
            "area が小さすぎる行が多い場合は min_area を上げます。",
            "area が極端に大きい行がある場合は max_area を下げるか split_touching を調整します。",
            "centroid_x / centroid_y は粒の中心位置です。画像上の座標確認に使えます。",
        ],
    )
    st.dataframe(result.measurements, use_container_width=True, hide_index=True)
    if not result.measurements.empty:
        st.bar_chart(result.measurements["area"], x_label="particle", y_label="area px")

with tabs[4]:
    st.write("現在のパラメータで検出した結果を保存します。CSVは測定表、TIFFは粒ごとに番号が入ったラベル画像です。")
    csv_data = result.measurements.to_csv(index=False).encode("utf-8-sig")
    label_data = labels_to_tiff_bytes(result.labels)
    st.download_button("測定CSVをダウンロード", csv_data, file_name="particle_measurements.csv", mime="text/csv")
    st.download_button("ラベルTIFFをダウンロード", label_data, file_name="particle_labels.tiff", mime="image/tiff")
