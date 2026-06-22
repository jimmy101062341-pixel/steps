"""
============================================================
 Step 0.5  手臂介面抽象層(arm-agnostic)
============================================================
roadmap 原則 1:手臂包成一個類別,pymycobot 只是內部實作。
上層程式只透過這個介面操作手臂,永遠不直接呼叫 pymycobot。
換手臂或上 ROS2 時,只需要再寫一個 ArmInterface 的子類別,上層不動。

刻意的設計決定:
- 介面一律用「公尺 + 4x4 齊次矩陣(pose)」 —— 對齊 ROS2 / MoveIt 的慣例。
- pymycobot 的 mm 與 rx/ry/rz(extrinsic xyz,已於 Step 0.2 驗證)只活在實作層,
  進出口各做一次轉換,上層完全看不到。
============================================================
"""

from abc import ABC, abstractmethod
import time
import numpy as np
from scipy.spatial.transform import Rotation


# ============================================================
#  抽象介面 —— 這層定義「手臂應該會做什麼」,與廠牌無關
# ============================================================
class ArmInterface(ABC):

    @abstractmethod
    def move_to_pose(self, pose: np.ndarray, speed: int = 30) -> bool:
        """移動 TCP 到目標 pose(4x4 齊次矩陣,公尺)。回傳是否成功。"""
        ...

    @abstractmethod
    def get_tcp_pose(self) -> np.ndarray:
        """回傳目前 TCP 的 4x4 齊次矩陣(公尺)。"""
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
#  共用小工具:pose 矩陣 <-> myCobot coords([x,y,z mm, rx,ry,rz deg])
#  (沿用 Step 0.2 驗好的 extrinsic xyz 慣例)
# ============================================================
def coords_to_pose(coords) -> np.ndarray:
    """myCobot coords(mm, deg) -> 4x4 pose(公尺)。"""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", coords[3:], degrees=True).as_matrix()
    T[:3, 3] = np.array(coords[:3]) / 1000.0      # mm -> m
    return T


def pose_to_coords(pose: np.ndarray):
    """4x4 pose(公尺) -> myCobot coords(mm, deg)。"""
    xyz_mm = (pose[:3, 3] * 1000.0).tolist()       # m -> mm
    rpy = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True).tolist()
    return xyz_mm + rpy


# ============================================================
#  pymycobot 實作 —— 廠牌相關的東西全關在這裡
# ============================================================
class MyCobotArm(ArmInterface):

    def __init__(self, port: str, baud: int = 115200):
        from pymycobot import MyCobot280            # 實作層才 import,上層不依賴它
        self._mc = MyCobot280(port, baud)
        time.sleep(1)
        self._mc.power_on()
        self._settle = 3.0                          # 等手臂停穩的秒數

    def move_to_pose(self, pose: np.ndarray, speed: int = 30) -> bool:
        coords = pose_to_coords(pose)
        self._mc.send_coords(coords, speed, 1)      # mode 1 = 直線插補
        time.sleep(self._settle)
        return True                                 # TODO: 之後可比對讀回值驗證到位

    def get_tcp_pose(self) -> np.ndarray:
        coords = self._mc.get_coords()
        if not coords:                              # 偶爾回 None/空,重讀一次
            time.sleep(0.2)
            coords = self._mc.get_coords()
        return coords_to_pose(coords)

    def grasp(self) -> bool:
        self._mc.set_gripper_state(1, 50)           # 1=關,速度 50
        time.sleep(1.5)
        return True

    def release(self) -> bool:
        self._mc.set_gripper_state(0, 50)           # 0=開
        time.sleep(1.5)
        return True


# ============================================================
#  示範:上層程式長這樣 —— 完全沒有 pymycobot 的影子
# ============================================================
if __name__ == "__main__":
    arm: ArmInterface = MyCobotArm("/dev/ttyUSB0", 115200)   # 換手臂只改這一行

    pose = arm.get_tcp_pose()
    print("目前 TCP pose(公尺):\n", np.round(pose, 4))

    # 在目前位置上方抬高 3cm(直接操作 pose 矩陣,單位是公尺)
    target = pose.copy()
    target[2, 3] += 0.03
    arm.move_to_pose(target)
    print("移動後:\n", np.round(arm.get_tcp_pose(), 4))

    # arm.grasp(); arm.release()      # 夾爪測試(確認 set_gripper_state 介面對得上再開)