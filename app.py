import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def srgb_to_linear(c):
    """sRGBのガンマ補正を解除してリニアRGBに変換"""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def rgb_to_xyz(rgb_linear):
    """リニアRGBをCIE XYZに変換 (D65光源)"""
    matrix = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ])
    return rgb_linear @ matrix.T

def get_spectrum_locus():
    """CIE 1931 スペクトル軌跡の標準データ (380nm-700nm)"""
    # 簡易的な等色関数近似データ (x, y)
    locus = [
        [0.1741, 0.0050], [0.1740, 0.0050], [0.1730, 0.0049], [0.1726, 0.0048],
        [0.1721, 0.0048], [0.1714, 0.0051], [0.1703, 0.0058], [0.1689, 0.0069],
        [0.1669, 0.0086], [0.1644, 0.0109], [0.1611, 0.0138], [0.1566, 0.0177],
        [0.1510, 0.0227], [0.1440, 0.0297], [0.1355, 0.0399], [0.1241, 0.0578],
        [0.1096, 0.0868], [0.0913, 0.1327], [0.0687, 0.2007], [0.0454, 0.2950],
        [0.0235, 0.4127], [0.0082, 0.5384], [0.0004, 0.6548], [0.0039, 0.7502],
        [0.0227, 0.8120], [0.0538, 0.8462], [0.0985, 0.8540], [0.1543, 0.8359],
        [0.2153, 0.8037], [0.2788, 0.7567], [0.3413, 0.7024], [0.4020, 0.6433],
        [0.4590, 0.5828], [0.5100, 0.5259], [0.5540, 0.4746], [0.5905, 0.4302],
        [0.6191, 0.3923], [0.6409, 0.3614], [0.6570, 0.3368], [0.6690, 0.3182],
        [0.6780, 0.3040], [0.6850, 0.2936], [0.6910, 0.2859], [0.6970, 0.2796],
        [0.7020, 0.2745], [0.7070, 0.2704], [0.7130, 0.2660], [0.7190, 0.2610],
        [0.7260, 0.2550], [0.7347, 0.2653]
    ]
    return np.array(locus)

def main():
    st.set_page_config(page_title="Chromaticity Plotter", layout="wide")
    st.title("🌈 CIE 1931 xy 色度図プロッター")
    st.write("画像の全ピクセルの色度を計算し、sRGB色域と比較します。")

    # サイドバーで設定
    st.sidebar.header("設定")
    sample_size = st.sidebar.slider("サンプル密度 (リサイズサイズ)", 50, 400, 150)
    alpha_val = st.sidebar.slider("プロットの透明度", 0.1, 1.0, 0.5)

    uploaded_file = st.file_uploader("画像をアップロード...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 画像読み込み
        img_raw = Image.open(uploaded_file).convert('RGB')
        
        # 処理用にリサイズ（高速化のため）
        img = img_raw.copy()
        img.thumbnail((sample_size, sample_size))
        
        img_array = np.array(img) / 255.0
        pixels = img_array.reshape(-1, 3)

        # 1. XYZ変換
        pixels_linear = srgb_to_linear(pixels)
        xyz = rgb_to_xyz(pixels_linear)

        # 2. xy色度計算
        sum_xyz = np.sum(xyz, axis=1, keepdims=True)
        sum_xyz[sum_xyz == 0] = 1e-9
        xy = xyz[:, :2] / sum_xyz

        # プロット作成
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # --- A. スペクトル軌跡 (馬蹄形の外枠) ---
        locus = get_spectrum_locus()
        ax.plot(locus[:, 0], locus[:, 1], color='black', linewidth=2, label="Spectrum Locus")
        # 始点と終点を結ぶ (純紫軌跡)
        ax.plot([locus[0,0], locus[-1,0]], [locus[0,1], locus[-1,1]], color='black', linewidth=2)

        # --- B. sRGB色域の三角形 ---
        # 頂点: R(0.64, 0.33), G(0.3, 0.6), B(0.15, 0.06)
        srgb_triangle = np.array([
            [0.64, 0.33], [0.30, 0.60], [0.15, 0.06], [0.64, 0.33]
        ])
        ax.plot(srgb_triangle[:, 0], srgb_triangle[:, 1], 'r--', linewidth=2, label="sRGB Gamut")

        # --- C. ピクセルデータのプロット ---
        ax.scatter(xy[:, 0], xy[:, 1], c=pixels, s=2, alpha=alpha_val, edgecolors='none', zorder=3)

        # --- D. D65白色点 ---
        ax.scatter([0.3127], [0.3290], color='black', marker='x', s=100, label='D65 White Point')
        ax.annotate("D65", (0.3127, 0.3290), textcoords="offset points", xytext=(5,5), fontsize=12)

        # グラフの整形
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.9)
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.set_title("CIE 1931 xy Chromaticity Diagram", fontsize=16)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        ax.set_facecolor('#f0f0f0')

        # 表示
        col1, col2 = st.columns([1, 1])
        with col1:
            st.pyplot(fig)
        with col2:
            st.image(img_raw, caption="元の画像", use_container_width=True)
            st.write(f"解析ピクセル数: {len(pixels)}")
            st.info("点の色は、そのピクセルの実際のRGB値に対応しています。三角形（sRGB）の外側にある点は、一般的なディスプレイでは正確に表現できない色（または計算上の色）であることを示します。")

if __name__ == "__main__":
    main()