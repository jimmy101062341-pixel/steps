"""
============================================================
 ArUco marker 產生工具(給 Step 0.4 用)
============================================================
產生可列印的 marker PNG。產生與偵測務必用同一個字典:DICT_4X4_50。

用法:
    python make_markers.py            # 產生 id 0,1,2 三顆,各含白邊
    python make_markers.py 0 5 7      # 產生指定 id

列印提醒:
- 印出來後「實際量一下黑色方塊的邊長(mm)」,記下來 —— 之後做 pose 估計要用真實邊長。
- 用一般紙即可,但盡量平整、不要反光;貼到硬板上更好(別貼彎)。
============================================================
"""

import sys
import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

MARKER_PX = 800     # marker 本體像素(印出來夠大、邊緣清楚)
BORDER_PX = 160     # 白邊(quiet zone)—— 一定要留,沒白邊偵測率大降


def make_one(marker_id: int):
    img = cv2.aruco.generateImageMarker(DICT, marker_id, MARKER_PX)
    canvas = np.full((MARKER_PX + 2 * BORDER_PX,
                      MARKER_PX + 2 * BORDER_PX), 255, dtype=np.uint8)
    canvas[BORDER_PX:BORDER_PX + MARKER_PX,
           BORDER_PX:BORDER_PX + MARKER_PX] = img
    # 在白邊下方標上 id,方便你分辨哪張是哪顆
    cv2.putText(canvas, f"id={marker_id}  DICT_4X4_50",
                (BORDER_PX, MARKER_PX + 2 * BORDER_PX - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
    fname = f"marker_{marker_id:02d}.png"
    cv2.imwrite(fname, canvas)
    print(f"已產生 {fname}")


if __name__ == "__main__":
    ids = [int(a) for a in sys.argv[1:]] or [0, 1, 2]
    for i in ids:
        make_one(i)
    print("\n列印後記得：量一下黑色方塊實際邊長(mm),寫下來。")