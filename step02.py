"""
============================================================
 Step 0.2  手臂連線 + 釘死 euler 角慣例
============================================================
目的:
  1) 確認 get_coords / send_coords 來回正常
  2) 單軸測試 -> 確認 rx/ry/rz 對應哪根實體軸 + 正負號
  3) 複合旋轉測試 -> 確認旋轉組合順序 (候選: extrinsic xyz)
  4) 提供並自我檢驗「座標 <-> 4x4 齊次矩陣」轉換函式

⚠ 安全須知(每次跑之前都看一眼):
  - 清空手臂周圍,給它完整活動空間,別放杯子/螢幕在揮動範圍內。
  - 速度先壓低(腳本用 speed=30)。手放在電源/急停旁。
  - 每個動作前腳本都會停下來等你按 Enter,先想一下手臂會往哪動再放行。
============================================================
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation

# pymycobot v3.6+ 用依型號分類的類別
from pymycobot import MyCobot280

# ---- 連線設定:依你的版本改這兩行 ----------------------------
# M5 版 (USB):  port 像 "COM3"(Windows)或 "/dev/ttyUSB0" / "/dev/ttyACM0"(Linux),baud 115200
# Pi 版 (內建): port "/dev/ttyAMA0",baud 1000000
PORT = "/dev/ttyAMA0"
BAUD = 1000000
# -------------------------------------------------------------

SPEED = 30      # 0-100,先慢
MODE  = 1       # 0=關節插補, 1=直線插補


import numpy as np
from scipy.spatial.transform import Rotation
from pymycobot import MyCobot280
import time

mc = MyCobot280(PORT, BAUD)   # 改成你的 port/baud

def coords_to_matrix(c):
    T = np.eye(4)
    T[:3,:3] = Rotation.from_euler("xyz", c[3:], degrees=True).as_matrix()
    T[:3,3] = c[:3]
    return T

def matrix_to_coords(T):
    rpy = Rotation.from_matrix(T[:3,:3]).as_euler("xyz", degrees=True)
    return [*T[:3,3], *rpy]

# 送一個帶旋轉的姿態,讀回,看你的函式還原得對不對
mc.send_coords([180, 0, 200, 0, 80, 0], 20, 1)
time.sleep(4)
actual = mc.get_coords()
back = matrix_to_coords(coords_to_matrix(actual))

print("手臂讀回:", [round(v,1) for v in actual])
print("轉一圈後:", [round(v,1) for v in back])
print("一致嗎? ", np.allclose(actual, back, atol=1.0))