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


def pause(msg=""):
    input(f"\n>> {msg}  [按 Enter 繼續]")


# ============================================================
#  轉換函式(候選慣例:extrinsic xyz == intrinsic ZYX == RPY)
#  R = Rz(rz) @ Ry(ry) @ Rx(rx)
# ============================================================
def coords_to_matrix(coords):
    """[x,y,z,rx,ry,rz] (mm, 度) -> 4x4 齊次矩陣"""
    x, y, z, rx, ry, rz = coords
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def matrix_to_coords(T):
    """4x4 齊次矩陣 -> [x,y,z,rx,ry,rz] (mm, 度)"""
    x, y, z = T[:3, 3]
    rx, ry, rz = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    return [float(x), float(y), float(z), float(rx), float(ry), float(rz)]


def selftest_roundtrip():
    """轉換函式自我一致性檢查(跟手臂無關,純數學)"""
    print("\n[自我檢查] coords -> matrix -> coords 是否還原...")
    for c in ([100, -50, 200, 30, 0, 0], [180, 0, 150, 45, -30, 90]):
        back = matrix_to_coords(coords_to_matrix(c))
        ok = np.allclose(c, back, atol=1e-6)
        print(f"  {c} -> {np.round(back, 3).tolist()}  {'OK' if ok else 'X'}")


# ============================================================
#  測試流程
# ============================================================
def main():
    print("連線中:", PORT, BAUD)
    mc = MyCobot280(PORT, BAUD)
    time.sleep(1)

    # --- 0) 基本連通 ---
    print("\n[0] 目前角度 get_angles():", mc.get_angles())
    print("    目前座標 get_coords():", mc.get_coords())
    pause("先送回一個安全的已知姿態(往下指、置中)")

    home = [180, 0, 200, 0, 0, 0]   # 視你的手臂工作範圍微調
    mc.send_coords(home, SPEED, MODE)
    time.sleep(3)
    c0 = mc.get_coords()
    print("    送出:", home)
    print("    讀回:", [round(v, 1) for v in c0])
    print("    (x,y,z 應該很接近;角度也應接近 0,0,0)")

    # --- 1) get/send 來回一致性 ---
    pause("[1] 來回測試:平移 +30mm 於 Z")
    target = list(c0)
    target[2] += 30
    mc.send_coords(target, SPEED, MODE)
    time.sleep(3)
    print("    送出 Z:", round(target[2], 1), " 讀回 Z:", round(mc.get_coords()[2], 1))

    # --- 2) 單軸旋轉:確認 軸對應 + 正負號 ---
    print("\n[2] 單軸旋轉測試 —— 仔細看手臂末端繞哪根軸轉、往哪邊轉,記下來。")
    for idx, name in [(3, "rx (預期繞 X / roll)"),
                      (4, "ry (預期繞 Y / pitch)"),
                      (5, "rz (預期繞 Z / yaw)")]:
        pause(f"    準備測 {name}:只把這一軸設成 +40 度")
        rot = list(home)
        rot[idx] = 40
        mc.send_coords(rot, SPEED, MODE)
        time.sleep(3)
        print(f"    送出: {rot}")
        print(f"    讀回: {[round(v,1) for v in mc.get_coords()]}")
        print(f"    >> 觀察:末端實際繞哪根實體軸轉?方向(順/逆)?  記到筆記。")
        mc.send_coords(home, SPEED, MODE)
        time.sleep(3)

    # --- 3) 複合旋轉:確認「順序」 ---
    print("\n[3] 複合旋轉測試 —— 釘死組合順序。")
    print("    在夾爪上貼一支筆/箭頭當指標,方便看朝向。")
    compound = list(home)
    compound[3] = 90   # rx
    compound[5] = 90   # rz   (ry=0,讓 Rx 與 Rz 的先後差異最明顯)
    R_xyz = Rotation.from_euler("xyz", [90, 0, 90], degrees=True).as_matrix()   # 候選 A:Rz·Ry·Rx
    R_zyx = Rotation.from_euler("zyx", [90, 0, 90], degrees=True).as_matrix()   # 候選 B:Rx·Ry·Rz
    print("    候選A(extrinsic xyz)預測末端 z 軸指向:", np.round(R_xyz[:, 2], 3))
    print("    候選B(extrinsic zyx)預測末端 z 軸指向:", np.round(R_zyx[:, 2], 3))
    print("    這兩個方向不同 —— 等下看手臂實際指哪個,就知道是哪個慣例。")
    pause("    送出複合旋轉 [rx=90, ry=0, rz=90]")
    mc.send_coords(compound, SPEED, MODE)
    time.sleep(3)
    print("    送出:", compound)
    print("    讀回:", [round(v, 1) for v in mc.get_coords()])
    print("    >> 末端指標方向比較接近 候選A 還是 候選B?  那個就是慣例。")

    pause("回 home,結束")
    mc.send_coords(home, SPEED, MODE)
    time.sleep(3)

    selftest_roundtrip()
    print("\n完成。把結論寫進筆記(見檔尾範本)。")


if __name__ == "__main__":
    main()


# ============================================================
#  筆記範本 —— 跑完把空格填上,這就是你之後永遠信任的依據
# ============================================================
"""
[myCobot 280 角度慣例 — 已驗證]
- 單位:度
- rx 繞 ____ 軸,+rx = ____(順/逆)時針
- ry 繞 ____ 軸,+ry = ____
- rz 繞 ____ 軸,+rz = ____
- 組合順序確認為:____ (extrinsic xyz=Rz·Ry·Rx / 其他: ____)
- 轉換用:Rotation.from_euler("____", [rx,ry,rz], degrees=True)
- 驗證日期 / 韌體版本:____
"""