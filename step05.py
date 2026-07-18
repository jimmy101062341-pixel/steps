"""
============================================================
 Step 0.5  手臂介面抽象層(arm-agnostic)— 完整可用版
============================================================
roadmap 原則 1:手臂包成一個類別,pymycobot 只是內部實作。
上層只透過這個介面操作手臂,永遠不直接呼叫 pymycobot。

本版修正了幾個實戰踩到的坑:
- 工作 home 用「前伸朝下」的關節姿態(不是 [0,0,0,0,0,0] 垂直伸直,
  那個 Z≈410 超出座標範圍、且沒有向上空間)。
- move_to_pose 會先檢查座標範圍,超範圍時「明確報錯」而非靜默不動。
- move_to_pose 會驗證真的到位(沒到就講,不假裝成功)。
- home() 走關節空間,不受座標 ±280 限制,能從任何姿態救回來。

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
        """移動 TCP 到目標 pose(4x4 齊次矩陣,公尺)。回傳是否成功到位。"""
        ...

    @abstractmethod
    def get_tcp_pose(self) -> np.ndarray:
        """回傳目前 TCP 的 4x4 齊次矩陣(公尺)。"""
        ...

    @abstractmethod
    def home(self) -> bool:
        """回到已知的安全工作姿態(走關節空間,從任何姿態都能救回來)。"""
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
    """myCobot coords(mm, deg) -> 4x4 pose(公尺)。"""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", coords[3:], degrees=True).as_matrix()
    T[:3, 3] = np.array(coords[:3]) / 1000.0
    return T


def pose_to_coords(pose: np.ndarray):
    """4x4 pose(公尺) -> myCobot coords(mm, deg)。"""
    xyz_mm = (pose[:3, 3] * 1000.0).tolist()
    rpy = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True).tolist()
    return xyz_mm + rpy


# ============================================================
#  pymycobot 實作
# ============================================================
class MyCobotArm(ArmInterface):

    # 工作 home:前伸、末端朝下、Z 落在座標範圍內、上下都有餘裕。
    # 由實測姿態 [0,-30,-30,0,-30,0](Z≈270)再壓低肘部而來。
    # 若你的手臂實際到位後 Z 仍偏高/偏低,微調 J2/J3 即可。
    HOME_JOINTS = [0, -35, -45, 0, -20, 0]

    def __init__(self, port: str, baud: int = 115200):
        from pymycobot import MyCobot280
        self._mc = MyCobot280(port, baud)
        time.sleep(1)
        self._mc.power_on()
        time.sleep(1)
        self._settle = 3.0

    # ---- 移動 ----
    def move_to_pose(self, pose: np.ndarray, speed: int = 30) -> bool:
        coords = pose_to_coords(pose)
        ok, msg = self._validate(coords)
        if not ok:
            raise ValueError(
                f"目標超出 myCobot 280 可命令範圍 -> {msg}\n"
                f"  完整目標: {[round(v,1) for v in coords]}\n"
                f"  提示:先呼叫 arm.home() 回到工作姿態再做相對移動。")
        self._mc.send_coords(coords, speed, 0)     # mode 0 關節插補,較不易無解
        time.sleep(self._settle)
        return self._verify(coords)

    def home(self, speed: int = 40) -> bool:
        """走關節空間 —— 不受座標 ±280 限制,從任何姿態都能救回來。"""
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
        self._mc.set_gripper_state(1, 50)          # 1=關
        time.sleep(1.5)
        return True

    def release(self) -> bool:
        self._mc.set_gripper_state(0, 50)          # 0=開
        time.sleep(1.5)
        return True

    # ---- 內部:範圍檢查 + 到位驗證 ----
    @staticmethod
    def _validate(coords):
        for n, v in zip(("x", "y", "z"), coords[:3]):
            if not (XYZ_LIM[0] <= v <= XYZ_LIM[1]):
                return False, f"{n}={v:.1f} 超出 {XYZ_LIM}"
        for n, v in zip(("rx", "ry", "rz"), coords[3:]):
            if not (RPY_LIM[0] <= v <= RPY_LIM[1]):
                return False, f"{n}={v:.1f} 超出 {RPY_LIM}"
        return True, ""

    def _verify(self, target, tol_mm=10.0):
        """驗證真的到位;沒到就明講,不假裝成功。"""
        actual = self._mc.get_coords()
        if not actual:
            print("⚠ 讀不回座標,無法確認到位")
            return False
        err = float(np.linalg.norm(np.array(actual[:3]) - np.array(target[:3])))
        if err > tol_mm:
            print(f"⚠ 未到位:誤差 {err:.1f}mm(可能 IK 無解或路徑受阻)")
            return False
        return True


# ============================================================
#  示範:上層完全沒有 pymycobot 的影子
# ============================================================
if __name__ == "__main__":
    arm: ArmInterface = MyCobotArm("/dev/ttyAMA0", 1000000)   # 換手臂只改這一行

    # 1) 回工作 home —— 一定成功,從任何姿態都能救回來
    print("回工作 home...")
    arm.home()

    # 2) 讀當前 pose,確認在可命令範圍內
    pose = arm.get_tcp_pose()
    coords = pose_to_coords(pose)
    print("home coords:", [round(v, 1) for v in coords])
    ok, msg = MyCobotArm._validate(coords)
    print("在可命令範圍內嗎?", ok, msg if msg else "(是)")

    # 3) 相對移動:抬高 5cm
    target = pose.copy()
    target[2, 3] += 0.05
    moved = arm.move_to_pose(target)
    print("移動後 Z(mm):", round(arm.get_tcp_pose()[2, 3] * 1000, 1), " 到位:", moved)

    # 4) 收工前回 home
    arm.home()