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

def get_precise_5nm_data():
    """ご提示いただいた5nm刻みの正確なCIE 1931 xyデータ"""
    return np.array([
        [380, 0.174112, 0.004964], [385, 0.174008, 0.004981], [390, 0.173801, 0.004915],
        [395, 0.17356, 0.004923], [400, 0.173337, 0.004797], [405, 0.173021, 0.004775],
        [410, 0.172577, 0.004799], [415, 0.172087, 0.004833], [420, 0.171407, 0.005102],
        [425, 0.170301, 0.005789], [430, 0.168878, 0.0069], [435, 0.166895, 0.008556],
        [440, 0.164412, 0.010858], [445, 0.161105, 0.013793], [450, 0.156641, 0.017705],
        [455, 0.150985, 0.02274], [460, 0.14396, 0.029703], [465, 0.135503, 0.039879],
        [470, 0.124118, 0.057803], [475, 0.109594, 0.086843], [480, 0.091294, 0.132702],
        [485, 0.068706, 0.200723], [490, 0.045391, 0.294976], [495, 0.02346, 0.412703],
        [500, 0.008168, 0.538423], [505, 0.003859, 0.654823], [510, 0.01387, 0.750186],
        [515, 0.038852, 0.812016], [520, 0.074302, 0.833803], [525, 0.114161, 0.826207],
        [530, 0.154722, 0.805864], [535, 0.192876, 0.781629], [540, 0.22962, 0.754329],
        [545, 0.265775, 0.724324], [550, 0.301604, 0.692308], [555, 0.337363, 0.658848],
        [560, 0.373102, 0.624451], [565, 0.408736, 0.589607], [570, 0.444062, 0.554714],
        [575, 0.478775, 0.520202], [580, 0.512486, 0.486591], [585, 0.544787, 0.454434],
        [590, 0.575151, 0.424232], [595, 0.602933, 0.396497], [600, 0.627037, 0.372491],
        [605, 0.648233, 0.351395], [610, 0.665764, 0.334011], [615, 0.680079, 0.319747],
        [620, 0.691504, 0.308342], [625, 0.700606, 0.299301], [630, 0.707918, 0.292027],
        [635, 0.714032, 0.285929], [640, 0.719033, 0.280935], [645, 0.723032, 0.276948],
        [650, 0.725992, 0.274008], [655, 0.728272, 0.271728], [660, 0.729969, 0.270031],
        [665, 0.731089, 0.268911], [670, 0.731993, 0.268007], [675, 0.732719, 0.267281],
        [680, 0.733417, 0.266583], [685, 0.734047, 0.265953], [690, 0.73439, 0.26561],
        [695, 0.734592, 0.265408], [700, 0.73469, 0.26531]
    ])

def main():
    st.set_page_config(page_title="High-Res Chromaticity Plotter", layout="wide")
    st.title("🔬 高精度(5nm) CIE 1931 xy 色度図プロッター")

    # サイドバー設定
    st.sidebar.header("解析設定")
    sample_res = st.sidebar.slider("解析解像度", 50, 600, 250)
    dot_alpha = st.sidebar.slider("プロット透明度", 0.01, 1.0, 0.3)
    dot_size = st.sidebar.slider("ドットサイズ", 1, 30, 3)
    label_step = st.sidebar.selectbox("波長ラベルの間隔", [10, 20, 50], index=1)

    uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        raw_img = Image.open(uploaded_file).convert('RGB')
        
        # 処理用にリサイズ
        img = raw_img.copy()
        img.thumbnail((sample_res, sample_res))
        
        # RGB正規化・フラット化
        pixels_rgb = np.array(img) / 255.0
        flat_pixels = pixels_rgb.reshape(-1, 3)

        # 変換
        pixels_linear = srgb_to_linear(flat_pixels)
        xyz = rgb_to_xyz(pixels_linear)
        sum_xyz = np.sum(xyz, axis=1, keepdims=True)
        sum_xyz[sum_xyz == 0] = 1e-9
        xy = xyz[:, :2] / sum_xyz

        # グラフ描画
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 1. 5nmスペクトル軌跡
        cie_data = get_precise_5nm_data()
        ax.plot(cie_data[:,1], cie_data[:,2], color='black', linewidth=1.5, zorder=3, label="Spectrum Locus")
        # 純紫軌跡
        ax.plot([cie_data[0,1], cie_data[-1,1]], [cie_data[0,2], cie_data[-1,2]], color='black', linewidth=1.5, zorder=3)

        # 2. 波長ラベル
        for val in cie_data:
            if val[0] % label_step == 0:
                ax.annotate(f"{int(val[0])}", (val[1], val[2]), textcoords="offset points", 
                            xytext=(5,5), fontsize=8, alpha=0.6)

        # 3. sRGB三角形
        srgb_tri = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06], [0.64, 0.33]])
        ax.plot(srgb_tri[:,0], srgb_tri[:,1], color='red', linestyle='--', linewidth=1, label="sRGB Gamut", zorder=4)

        # 4. 全ピクセルのラスタープロット
        ax.scatter(xy[:, 0], xy[:, 1], c=flat_pixels, s=dot_size, alpha=dot_alpha, edgecolors='none', zorder=2)

        # 5. D65白色点
        ax.scatter([0.3127], [0.3290], color='black', marker='+', s=100, label='D65', zorder=5)

        # グラフ設定
        ax.set_xlim(0, 0.85)
        ax.set_ylim(0, 0.9)
        ax.set_aspect('equal')
        ax.set_xlabel('CIE x')
        ax.set_ylabel('CIE y')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend()
        ax.set_title("CIE 1931 xy Chromaticity Diagram (5nm resolution)")

        # レイアウト表示
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.pyplot(fig)
        with col2:
            st.image(raw_img, caption="入力画像", use_container_width=True)
            st.write(f"解析プロット点数: {len(flat_pixels)}")
            st.markdown("""
            ### 解析のヒント
            - **色域外の色**: 赤い破線(sRGB)の外側にある点は、一般的なモニターでは本来の色を再現できない領域です。
            - **分布の密度**: プロットが集中している場所が、その画像の支配的な色相を示します。
            - **透明度の調整**: サイドバーの「プロット透明度」を下げると、データの重なり（密度）がより明確に見えます。
            """)

if __name__ == "__main__":
    main()