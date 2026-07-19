"""
============================================================
 手臂活動範圍探索(安全優先)
============================================================
目的:摸清這台手臂「安全可達」的範圍,供之後規劃架子/站點高度。
兩種掃描:
  A. 關節空間掃描 —— 每軸在「保守中段範圍」轉一輪,看末端掃出的 XYZ
  B. 座標空間掃描 —— 在 XYZ 網格上逐點探,記錄能安全到達的點

防手臂自我卡住/纏繞:
  - 關節掃描只走保守中段(不到 ±極限,避開最易纏繞/奇異的邊緣)
  - 每點都經 validate(±280) + 到位檢查,搆不到就跳過記錄,絕不硬頂
防撞桌:座標掃描設 Z_MIN 下限,低於此高度不去。

★ 現況:裸手臂、無夾爪、末端相機尚未掛。
  之後掛上末端 OAK 後「重跑這支」對比 —— 相機會讓末端變長變重,可達範圍會縮。
  固定那台相機在手臂搆不到處,故不納入碰撞考量。

安全:全程慢速,手放急停旁。第一次跑先站著看,異常立刻停。
============================================================
"""
import sys
import time
import json
import numpy as np
from pymycobot import MyCobot280

PORT, BAUD = "/dev/ttyAMA0", 1000000
SPEED = 25                       # 慢速
SETTLE = 2.5

# --- 安全盒(mm)--- 依你的桌面/環境調
Z_MIN = 120.0                    # 別低於此高度(避免撞桌;裸手臂可再低,保守起見)
Z_MAX = 260.0                    # 別高於此(留餘裕,雖然固定相機搆不到)
XY_MAX_R = 260.0                 # 末端水平距底座的最大半徑(reach 280,留餘裕)

# --- 關節保守中段範圍(度)--- 只掃中段,避開纏繞/奇異的極限
# myCobot 280 各軸極限約 ±165,這裡刻意收窄
JOINT_SCAN = {
    0: [-90, -45, 0, 45, 90],        # J1 底座旋轉
    1: [-60, -35, -10],              # J2 大臂
    2: [-70, -45, -20],              # J3 肘
    3: [-45, 0, 45],                 # J4 腕轉
    4: [-40, -10, 20],               # J5 腕俯仰
    5: [0],                          # J6 末端自轉(對範圍探索影響小,固定)
}

HOME = [0, -35, -60, 0, -10, 0]      # Step 0.5 定案的中段 home


def connect():
    mc = MyCobot280(PORT, BAUD)
    time.sleep(1); mc.power_on(); time.sleep(1)
    return mc


def go_home(mc):
    mc.send_angles(HOME, 40); time.sleep(5)


def read_xyz(mc):
    c = mc.get_coords()
    return np.array(c[:3]) if c else None


def in_safe_box(xyz):
    x, y, z = xyz
    if not (Z_MIN <= z <= Z_MAX):
        return False, f"Z={z:.0f} 超出盒[{Z_MIN:.0f},{Z_MAX:.0f}]"
    if np.hypot(x, y) > XY_MAX_R:
        return False, f"半徑 {np.hypot(x,y):.0f} > {XY_MAX_R:.0f}"
    return True, ""


# ============================================================
#  A. 關節空間掃描
# ============================================================
def scan_joints(mc):
    print("\n" + "=" * 55)
    print(" A. 關節空間掃描(保守中段,防纏繞)")
    print("=" * 55)
    reached = []
    # 掃 J1~J5,每次只動一軸、其餘保持 home,避免多軸同時到極限而卡住
    base = list(HOME)
    for axis, vals in JOINT_SCAN.items():
        if len(vals) <= 1:
            continue
        print(f"\n-- 掃 J{axis+1} --")
        for v in vals:
            angles = list(base)
            angles[axis] = v
            mc.send_angles(angles, SPEED)
            time.sleep(SETTLE)
            xyz = read_xyz(mc)
            if xyz is None:
                print(f"  J{axis+1}={v:>4}  讀取失敗,跳過")
                continue
            safe, why = in_safe_box(xyz)
            tag = "✓" if safe else f"⚠越界({why})"
            print(f"  J{axis+1}={v:>4}  末端 XYZ=[{xyz[0]:6.0f},{xyz[1]:6.0f},{xyz[2]:6.0f}]  {tag}")
            if safe:
                reached.append(xyz.tolist())
        mc.send_angles(base, SPEED); time.sleep(SETTLE)   # 該軸掃完回中位
    return reached


# ============================================================
#  B. 座標空間掃描(XYZ 網格)
# ============================================================
def scan_coords(mc):
    print("\n" + "=" * 55)
    print(" B. 座標空間掃描(XYZ 網格,防撞桌)")
    print("=" * 55)
    go_home(mc)
    home_c = mc.get_coords()
    if not home_c:
        print("讀不到 home 座標,略過座標掃描"); return []
    rx, ry, rz = home_c[3:]        # 掃描時維持 home 的末端姿態

    # 網格:在安全盒內取樣
    xs = np.arange(140, 241, 50)   # X
    ys = np.arange(-120, 121, 60)  # Y
    zs = np.arange(Z_MIN, Z_MAX + 1, 40)  # Z

    reached = []
    cur = np.array(home_c[:3], dtype=float)
    for z in zs:
        for x in xs:
            for y in ys:
                target = [x, y, z]
                safe, why = in_safe_box(np.array(target))
                if not safe:
                    continue
                # 用小步閉環靠近(沿用 Step 0.5 精神,搆不到就跳過)
                ok = _closed_loop_to(mc, target + [rx, ry, rz])
                a = read_xyz(mc)
                if ok and a is not None:
                    err = np.linalg.norm(a - np.array(target))
                    print(f"  目標[{x:4.0f},{y:4.0f},{z:4.0f}] -> "
                          f"到[{a[0]:4.0f},{a[1]:4.0f},{a[2]:4.0f}] 誤差{err:4.0f}  ✓")
                    reached.append(a.tolist())
                else:
                    print(f"  目標[{x:4.0f},{y:4.0f},{z:4.0f}] -> 搆不到 ✗")
    return reached


def _closed_loop_to(mc, target, gain=0.7, tol=10, min_impr=2, max_it=5):
    """簡化版閉環,搆不到回 False。"""
    target = np.array(target, dtype=float)
    if not (-280 <= target[:3]).all() or not (target[:3] <= 280).all():
        return False
    prev = None
    for _ in range(max_it):
        c = mc.get_coords()
        if not c:
            return False
        cur = np.array(c, dtype=float)
        err = np.linalg.norm((target - cur)[:3])
        if err < tol:
            return True
        if prev is not None and (prev - err) < min_impr:
            return False
        prev = err
        cmd = cur + gain * (target - cur)
        cmd[:3] = np.clip(cmd[:3], -280, 280)
        mc.send_coords(cmd.tolist(), SPEED, 0)
        time.sleep(SETTLE)
    return False


def summarize(joint_pts, coord_pts):
    print("\n" + "=" * 55)
    print(" 可達範圍總結")
    print("=" * 55)
    allpts = np.array(joint_pts + coord_pts)
    if len(allpts) == 0:
        print("沒有可達點。"); return
    lo, hi = allpts.min(0), allpts.max(0)
    print(f"  安全可達 X: {lo[0]:6.0f} ~ {hi[0]:6.0f} mm")
    print(f"  安全可達 Y: {lo[1]:6.0f} ~ {hi[1]:6.0f} mm")
    print(f"  安全可達 Z: {lo[2]:6.0f} ~ {hi[2]:6.0f} mm")
    print(f"  可達點總數: {len(allpts)}")
    json.dump({"joint_pts": joint_pts, "coord_pts": coord_pts,
               "x_range": [float(lo[0]), float(hi[0])],
               "y_range": [float(lo[1]), float(hi[1])],
               "z_range": [float(lo[2]), float(hi[2])]},
              open("workspace_map.json", "w"), indent=2)
    print("\n已存 workspace_map.json —— 之後規劃架子/站點高度用這個。")
    print("提醒:掛上末端相機後重跑,可達範圍會縮,兩份對比即知相機吃掉多少空間。")


if __name__ == "__main__":
    print(__doc__)
    input("確認手臂周圍淨空、手放急停旁,按 Enter 開始...")
    mc = connect()
    go_home(mc)

    jp = scan_joints(mc)
    go_home(mc)
    cp = scan_coords(mc)
    go_home(mc)

    summarize(jp, cp)
    print("\n回 home,結束。")