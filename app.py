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
    # rgb_linearは (N, 3) の形状を想定
    return rgb_linear @ matrix.T

def draw_cie_boundary():
    """xy色度図の外形（馬蹄形）の境界線を描画するためのデータを返す"""
    # 簡略化のため、色度図の枠線用データを用意
    # 実際には波長ごとの等色関数から計算が必要ですが、ここでは一般的な形状を近似します
    from matplotlib.patches import Polygon
    # 等色関数データ（簡易版：可視光範囲のxy座標）
    spectrum_xy = [
        [0.1741, 0.0050], [0.1740, 0.0050], [0.1730, 0.0049], [0.1726, 0.0048],
        [0.1700, 0.0055], [0.1600, 0.0200], [0.1200, 0.1000], [0.0500, 0.3000],
        [0.0100, 0.6000], [0.1000, 0.8000], [0.2000, 0.8000], [0.4000, 0.6000],
        [0.6000, 0.3500], [0.7000, 0.2800], [0.7347, 0.2653]
    ]
    # 本来はもっと詳細なデータが必要ですが、背景として目安を表示します
    return spectrum_xy

def main():
    st.title("Image to xy Chromaticity Plotter")
    st.write("画像をアップロードすると、全ピクセルの色度をCIE 1931 xy図にプロットします。")

    uploaded_file = st.file_uploader("画像を選択してください...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 画像読み込み
        img = Image.open(uploaded_file).convert('RGB')
        
        # 処理を軽くするためにリサイズ（ピクセル数が多いとプロットが重くなるため）
        max_size = 200 
        img.thumbnail((max_size, max_size))
        
        img_array = np.array(img) / 255.0
        h, w, _ = img_array.shape
        pixels = img_array.reshape(-1, 3)

        # 1. リニアRGBへ変換
        pixels_linear = srgb_to_linear(pixels)

        # 2. XYZへ変換
        xyz = rgb_to_xyz(pixels_linear)

        # 3. xy色度に変換
        # X + Y + Z = 0 の場合のゼロ除算を防ぐ
        sum_xyz = np.sum(xyz, axis=1, keepdims=True)
        sum_xyz[sum_xyz == 0] = 1e-9
        xy = xyz[:, :2] / sum_xyz

        # プロット作成
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 背景のxy色度図のガイド（簡易表示）
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 0.9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linestyle=':', alpha=0.6)

        # 全ピクセルを散布図としてプロット
        # colors引数に元のRGB値を渡すことで、点の色をそのピクセルの色にする
        ax.scatter(xy[:, 0], xy[:, 1], c=pixels, s=1, alpha=0.5, edgecolors='none')

        # D65白色点のプロット
        ax.scatter([0.3127], [0.3290], color='black', marker='x', label='D65 White Point')
        
        st.pyplot(fig)
        
        st.write(f"処理ピクセル数: {len(pixels)} ピクセル")
        st.image(img, caption="解析対象画像 (リサイズ済)", use_column_width=False)

if __name__ == "__main__":
    main()