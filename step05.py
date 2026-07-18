"""
============================================================
 Step 0.5  手臂介面抽象層(arm-agnostic)— 完整可用版 v2
============================================================
roadmap 原則 1:手臂包成一個類別,pymycobot 只是內部實作。
上層只透過這個介面操作手臂,永遠不直接呼叫 pymycobot。

實戰修正:
- 工作 home 用「前伸朝下」關節姿態(非 [0,0,0,0,0,0],那個 Z≈410 超範圍)。
- move_to_pose 大位移「自動拆成小步」——這支手臂大幅座標移動會嚴重打折/易無解,
  實測送 20mm 穩定會動,送 50mm 只到約 30mm。拆步後每步都在可動範圍內。
- 到位容忍度設 25mm:誠實反映這支手臂座標控制的實際精度(約 1-2cm)。
  ★ 重要結論:280 裸座標精度不足以「裸插」試管,最後對位必須靠
    hand-eye 相機閉環 + 機構導引(漏斗/V槽)。move_to_pose 只負責「帶到附近」。
- move_to_pose 超範圍時明確報錯;home() 走關節空間,能從任何姿態救回來。

介面單位:公尺 + 4x4 齊次矩陣(對齊 ROS2/MoveIt)。
pymycobot 的 mm 與 rx/ry/rz(extrinsic xyz,Step 0.2 驗證)只活在實作層。
============================================================
"""

from abc import ABC, abstractmethod
import time
import numpy as np
from scipy.spatial.transform import Rotation

# myCobot 280 send_coords 有效輸入範圍(官方文件:xyz ±280, rpy ±314)
XYZ_LIM = (-280.0, 280.0)
RPY_LIM = (-314.0, 314.0)


# ============================================================
#  抽象介面 —— 與廠牌無關
# ============================================================
class ArmInterface(ABC):

    @abstractmethod
    def move_to_pose(self, pose: np.ndarray, speed: int = 30) -> bool:
        """移動 TCP 到目標 pose(4x4 齊次矩陣,公尺)。回傳是否到位。"""
        ...

    @abstractmethod
    def get_tcp_pose(self) -> np.ndarray:
        """回傳目前 TCP 的 4x4 齊次矩陣(公尺)。"""
        ...

    @abstractmethod
    def home(self) -> bool:
        """回到已知安全工作姿態(走關節空間,從任何姿態都能救回來)。"""
        ...

    @abstractmethod
    def grasp(self) -> bool:
        """閉合夾爪。"""
        ...

    @abstractmethod
    def release(self) -> bool:
        """張開夾爪。"""
        ...


# ============================================================
#  pose 矩陣 <-> myCobot coords(沿用 Step 0.2 的 extrinsic xyz)
# ============================================================
def coords_to_pose(coords) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", coords[3:], degrees=True).as_matrix()
    T[:3, 3] = np.array(coords[:3]) / 1000.0
    return T


def pose_to_coords(pose: np.ndarray):
    xyz_mm = (pose[:3, 3] * 1000.0).tolist()
    rpy = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True).tolist()
    return xyz_mm + rpy


# ============================================================
#  pymycobot 實作
# ============================================================
class MyCobotArm(ArmInterface):

    # 工作 home:前伸、末端朝下、Z≈197(範圍內、上下有餘裕)。實測定案。
    HOME_JOINTS = [0, -35, -45, 0, -20, 0]

    def __init__(self, port: str, baud: int = 115200):
        from pymycobot import MyCobot280
        self._mc = MyCobot280(port, baud)
        time.sleep(1)
        self._mc.power_on()
        time.sleep(1)
        self._settle = 3.0

    # ---- 移動(大位移自動拆小步)----
    def move_to_pose(self, pose: np.ndarray, speed: int = 30,
                     max_step_mm: float = 20.0) -> bool:
        target = pose_to_coords(pose)
        ok, msg = self._validate(target)
        if not ok:
            raise ValueError(
                f"目標超出 myCobot 280 可命令範圍 -> {msg}\n"
                f"  完整目標: {[round(v,1) for v in target]}\n"
                f"  提示:先呼叫 arm.home() 回工作姿態再做相對移動。")

        start = self._mc.get_coords()
        if not start:
            print("⚠ 讀不到起點座標")
            return False

        # 依直線距離把移動拆成 <= max_step_mm 的小步
        dist = float(np.linalg.norm(np.array(target[:3]) - np.array(start[:3])))
        n = max(1, int(np.ceil(dist / max_step_mm)))
        for k in range(1, n + 1):
            inter = [start[i] + (target[i] - start[i]) * k / n for i in range(6)]
            self._mc.send_coords(inter, speed, 0)     # mode 0 關節插補
            time.sleep(self._settle if k == n else 1.0)

        return self._verify(target)

    def home(self, speed: int = 40) -> bool:
        self._mc.send_angles(list(self.HOME_JOINTS), speed)
        time.sleep(5)
        c = self._mc.get_coords()
        if c:
            print(f"home 到位,coords = {[round(v,1) for v in c]}")
        return True

    # ---- 讀取 ----
    def get_tcp_pose(self) -> np.ndarray:
        for _ in range(3):
            coords = self._mc.get_coords()
            if coords:
                return coords_to_pose(coords)
            time.sleep(0.2)
        raise RuntimeError("get_coords 連續讀取失敗")

    # ---- 夾爪 ----
    def grasp(self) -> bool:
        self._mc.set_gripper_state(1, 50)
        time.sleep(1.5)
        return True

    def release(self) -> bool:
        self._mc.set_gripper_state(0, 50)
        time.sleep(1.5)
        return True

    # ---- 內部 ----
    @staticmethod
    def _validate(coords):
        for n, v in zip(("x", "y", "z"), coords[:3]):
            if not (XYZ_LIM[0] <= v <= XYZ_LIM[1]):
                return False, f"{n}={v:.1f} 超出 {XYZ_LIM}"
        for n, v in zip(("rx", "ry", "rz"), coords[3:]):
            if not (RPY_LIM[0] <= v <= RPY_LIM[1]):
                return False, f"{n}={v:.1f} 超出 {RPY_LIM}"
        return True, ""

    def _verify(self, target, tol_mm=25.0):
        """到位驗證。25mm 容忍度誠實反映這支手臂座標控制的實際精度。"""
        actual = self._mc.get_coords()
        if not actual:
            print("⚠ 讀不回座標,無法確認到位")
            return False
        err = float(np.linalg.norm(np.array(actual[:3]) - np.array(target[:3])))
        if err > tol_mm:
            print(f"⚠ 未完全到位:誤差 {err:.1f}mm")
            return False
        print(f"  到位(誤差 {err:.1f}mm)")
        return True


# ============================================================
#  示範:上層完全沒有 pymycobot 的影子
# ============================================================
if __name__ == "__main__":
    arm: ArmInterface = MyCobotArm("/dev/ttyAMA0", 1000000)   # 換手臂只改這一行

    print("回工作 home...")
    arm.home()

    pose = arm.get_tcp_pose()
    coords = pose_to_coords(pose)
    print("home coords:", [round(v, 1) for v in coords])
    ok, msg = MyCobotArm._validate(coords)
    print("在可命令範圍內嗎?", ok, msg if msg else "(是)")

    # 抬高 5cm —— 會自動拆成小步
    target = pose.copy()
    target[2, 3] += 0.05
    moved = arm.move_to_pose(target)
    print("移動後 Z(mm):", round(arm.get_tcp_pose()[2, 3] * 1000, 1), " 到位:", moved)

    arm.home()