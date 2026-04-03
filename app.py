import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def srgb_to_linear(c):
    """sRGBのガンマ補正を解除してリニアRGBに変換"""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def rgb_to_xyz(rgb_linear):
    """リニアRGBをCIE XYZに変換 (D65光源用の標準変換行列)"""
    matrix = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ])
    return rgb_linear @ matrix.T

def get_precise_cie_data():
    """ご指定いただいた10nm刻みの正確なCIE 1931 xyデータ"""
    # [wavelength, x, y]
    data = [
        [380, 0.174112, 0.004964], [390, 0.173801, 0.004915], [400, 0.173337, 0.004797],
        [410, 0.172577, 0.004799], [420, 0.171407, 0.005102], [430, 0.168878, 0.0069],
        [440, 0.164412, 0.010858], [450, 0.156641, 0.017705], [460, 0.14396, 0.029703],
        [470, 0.124118, 0.057803], [480, 0.091294, 0.132702], [490, 0.045391, 0.294976],
        [500, 0.008168, 0.538423], [510, 0.01387, 0.750186], [520, 0.074302, 0.833803],
        [530, 0.154722, 0.805864], [540, 0.22962, 0.754329], [550, 0.301604, 0.692308],
        [560, 0.373102, 0.624451], [570, 0.444062, 0.554714], [580, 0.512486, 0.486591],
        [590, 0.575151, 0.424232], [600, 0.627037, 0.372491], [610, 0.665764, 0.334011],
        [620, 0.691504, 0.308342], [630, 0.707918, 0.292027], [640, 0.719033, 0.280935],
        [650, 0.725992, 0.274008], [660, 0.729969, 0.270031], [670, 0.731993, 0.268007],
        [680, 0.733417, 0.266583], [690, 0.73439, 0.26561], [700, 0.73469, 0.26531]
    ]
    return np.array(data)

def main():
    st.set_page_config(page_title="XY Chromaticity Plotter", layout="wide")
    st.title("🧪 高精度 xy 色度図解析ツール")

    # サイドバー設定
    st.sidebar.header("表示オプション")
    sample_res = st.sidebar.slider("解析解像度 (px)", 50, 500, 200, help="値を大きくすると詳細になりますが処理が重くなります")
    dot_alpha = st.sidebar.slider("ドットの透明度", 0.01, 1.0, 0.2)
    dot_size = st.sidebar.slider("ドットの大きさ", 1, 20, 2)
    show_wavelength = st.sidebar.checkbox("波長数値を表示", True)

    uploaded_file = st.file_uploader("解析したい画像をアップロード", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 画像読み込み
        raw_img = Image.open(uploaded_file).convert('RGB')
        
        # 処理用にリサイズ
        img = raw_img.copy()
        img.thumbnail((sample_res, sample_res))
        
        # RGB正規化 (0-1)
        pixels_rgb = np.array(img) / 255.0
        flat_pixels = pixels_rgb.reshape(-1, 3)

        # 1. 変換: sRGB -> Linear RGB -> XYZ -> xy
        pixels_linear = srgb_to_linear(flat_pixels)
        xyz = rgb_to_xyz(pixels_linear)
        sum_xyz = np.sum(xyz, axis=1, keepdims=True)
        sum_xyz[sum_xyz == 0] = 1e-9  # ゼロ除算回避
        xy = xyz[:, :2] / sum_xyz

        # プロット作成
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # スペクトル軌跡の描画 (ご指定の数値)
        cie_data = get_precise_cie_data()
        wl, lx, ly = cie_data[:,0], cie_data[:,1], cie_data[:,2]
        
        # 軌跡（外枠）
        ax.plot(lx, ly, color='black', linewidth=2, label="Spectrum Locus", zorder=3)
        # 純紫軌跡（380nmと700nmを直線で結ぶ）
        ax.plot([lx[0], lx[-1]], [ly[0], ly[-1]], color='black', linewidth=2, zorder=3)

        # 波長ラベルの描画
        if show_wavelength:
            for i, val in enumerate(cie_data):
                if val[0] % 20 == 0: # 20nmごとに表示
                    ax.annotate(f"{int(val[0])}", (val[1], val[2]), textcoords="offset points", 
                                xytext=(5,5), fontsize=8, alpha=0.7)

        # sRGB色域の三角形
        srgb_tri = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06], [0.64, 0.33]])
        ax.plot(srgb_tri[:,0], srgb_tri[:,1], color='red', linestyle='--', linewidth=1, label="sRGB Gamut", zorder=4)

        # 全ピクセルのプロット (ラスタープロット)
        ax.scatter(xy[:, 0], xy[:, 1], c=flat_pixels, s=dot_size, alpha=dot_alpha, edgecolors='none', zorder=2)

        # D65白色点
        ax.scatter([0.3127], [0.3290], color='black', marker='+', s=100, label='D65', zorder=5)

        # グラフ装飾
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.9)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend()
        ax.set_title("CIE 1931 Chromaticity Diagram")

        # Streamlit表示
        col1, col2 = st.columns([2, 1])
        with col1:
            st.pyplot(fig)
        with col2:
            st.image(raw_img, caption="元の画像", use_container_width=True)
            st.write(f"解析サンプル数: {len(flat_pixels)} 点")
            st.info("※解析を高速化するため、画像をリサイズしてプロットしています。サイドバーの「解析解像度」で精度を調整できます。")

if __name__ == "__main__":
    main()