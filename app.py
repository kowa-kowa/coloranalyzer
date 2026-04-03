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

def get_cie_1931_data():
    """CIE 1931 2-deg, 10nm刻みの等色関数から計算したxy座標データ"""
    # (wavelength, x, y)
    data = [
        (380, 0.1741, 0.0050), (390, 0.1738, 0.0049), (400, 0.1733, 0.0048),
        (410, 0.1726, 0.0048), (420, 0.1714, 0.0051), (430, 0.1689, 0.0069),
        (440, 0.1644, 0.0109), (450, 0.1566, 0.0177), (460, 0.1440, 0.0297),
        (470, 0.1241, 0.0578), (480, 0.0913, 0.1327), (490, 0.0454, 0.2950),
        (500, 0.0047, 0.5384), (510, 0.0139, 0.7502), (520, 0.1176, 0.8339),
        (530, 0.2153, 0.8037), (540, 0.3173, 0.7185), (550, 0.4120, 0.6121),
        (560, 0.4991, 0.5002), (570, 0.5735, 0.4260), (580, 0.6321, 0.3676),
        (590, 0.6757, 0.3242), (600, 0.7048, 0.2952), (610, 0.7230, 0.2769),
        (620, 0.7328, 0.2671), (630, 0.7341, 0.2656), (640, 0.7344, 0.2654),
        (650, 0.7346, 0.2653), (660, 0.7347, 0.2653), (670, 0.7347, 0.2653),
        (680, 0.7347, 0.2653), (690, 0.7347, 0.2653), (700, 0.7347, 0.2653)
    ]
    return np.array(data)

def main():
    st.set_page_config(page_title="Accurate Chromaticity Plotter", layout="wide")
    st.title("🎨 精密 CIE 1931 xy 色度図プロッター")
    
    st.sidebar.header("表示設定")
    sample_size = st.sidebar.slider("サンプル密度 (リサイズ)", 50, 400, 150)
    alpha_val = st.sidebar.slider("プロットの透明度", 0.05, 1.0, 0.3)
    show_labels = st.sidebar.checkbox("波長ラベルを表示", True)

    uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img_raw = Image.open(uploaded_file).convert('RGB')
        img = img_raw.copy()
        img.thumbnail((sample_size, sample_size))
        
        img_array = np.array(img) / 255.0
        pixels = img_array.reshape(-1, 3)

        # 変換処理
        pixels_linear = srgb_to_linear(pixels)
        xyz = rgb_to_xyz(pixels_linear)
        sum_xyz = np.sum(xyz, axis=1, keepdims=True)
        sum_xyz[sum_xyz == 0] = 1e-9
        xy = xyz[:, :2] / sum_xyz

        # プロット
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 1. スペクトル軌跡 (正確な10nm刻みデータ)
        cie_data = get_cie_1931_data()
        wavelengths = cie_data[:, 0]
        locus_x = cie_data[:, 1]
        locus_y = cie_data[:, 2]
        
        # 軌跡の線を描画
        ax.plot(locus_x, locus_y, color='#333333', linewidth=2, label="Spectrum Locus", zorder=4)
        # 純紫軌跡 (380nmと700nmを結ぶ)
        ax.plot([locus_x[0], locus_x[-1]], [locus_y[0], locus_y[-1]], color='#333333', linewidth=2, zorder=4)

        # 波長ラベルの追加 (20nm刻みで表示)
        if show_labels:
            for w, x, y in cie_data:
                if w % 20 == 0:
                    ax.annotate(f"{int(w)}", (x, y), textcoords="offset points", xytext=(0,10), 
                                fontsize=9, color='blue', ha='center')

        # 2. sRGB色域三角形
        srgb_triangle = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06], [0.64, 0.33]])
        ax.plot(srgb_triangle[:, 0], srgb_triangle[:, 1], 'r--', linewidth=1.5, label="sRGB Gamut", zorder=5)

        # 3. ピクセルデータのプロット
        ax.scatter(xy[:, 0], xy[:, 1], c=pixels, s=5, alpha=alpha_val, edgecolors='none', zorder=2)

        # 4. D65白色点
        ax.scatter([0.3127], [0.3290], color='black', marker='+', s=100, label='D65 White Point', zorder=6)

        # グラフの仕上げ
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.9)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title("CIE 1931 xy Chromaticity Diagram (Accurate 10nm Data)", fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper right')
        ax.set_facecolor('#fafafa')

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.pyplot(fig)
        with col2:
            st.image(img_raw, caption="Uploaded Image", use_container_width=True)
            st.write(f"処理ピクセル数: {len(pixels)}")
            st.markdown("""
            ### グラフの見方
            - **黒い曲線 (Spectrum Locus)**: 人間が見ることができる色の限界（単色光）です。
            - **赤い破線三角形 (sRGB Gamut)**: 一般的なモニター(sRGB)が表現できる色の範囲です。
            - **カラー点**: 画像内のピクセルの色度です。三角形からはみ出ている点は、sRGBモニターでは正しく表示できない「非常に鮮やかな色」であることを意味します。
            """)

if __name__ == "__main__":
    main()