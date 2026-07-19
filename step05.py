"""
============================================================
 Step 0.5  手臂介面抽象層(arm-agnostic)— v3 閉環版
============================================================
roadmap 原則 1:手臂包成一個類別,pymycobot 只是內部實作。
上層只透過這個介面操作手臂,永遠不直接呼叫 pymycobot。

v3 重點:move_to_pose 改用「閉環比例控制」取代固定拆步 ——
  讀當前 -> 算誤差(目標−實際)-> 往誤差方向送 gain 倍 -> 重複到夠近。
  這比開環拆步聰明:自動適應姿態相關的打折,誤差大就多修幾次。
  (注意:這是閉環比例控制,不是 gradient descent —— 誤差方向已知,不需摸索。)

閉環三道守門(validate 的完整角色):
  1. 範圍檢查:每次迭代送出前檢查 ±280,超範圍 clamp 回邊界
  2. 收斂檢查:誤差 < tolerance 就完成
  3. 卡住檢查:連續改善 < min_improvement 就判定到極限,誠實中止

★ 定位:此閉環把座標精度從 ~1-2cm 逼到 ~mm 級,但收不到比手臂本身更準。
  插試管需 sub-mm,仍須靠 hand-eye 相機閉環 + 機構導引。move_to_pose 只「帶到附近」。

介面單位:公尺 + 4x4 齊次矩陣(對齊 ROS2/MoveIt)。
euler 慣例 extrinsic xyz(Step 0.2 驗證)。
============================================================
"""

from abc import ABC, abstractmethod
import time
import numpy as np
from scipy.spatial.transform import Rotation

XYZ_LIM = (-280.0, 280.0)
RPY_LIM = (-314.0, 314.0)


# ============================================================
#  抽象介面
# ============================================================
class ArmInterface(ABC):

    @abstractmethod
    def move_to_pose(self, pose: np.ndarray, speed: int = 30) -> bool: ...

    @abstractmethod
    def get_tcp_pose(self) -> np.ndarray: ...

    @abstractmethod
    def home(self) -> bool: ...

    @abstractmethod
    def grasp(self) -> bool: ...

    @abstractmethod
    def release(self) -> bool: ...


# ============================================================
#  pose 矩陣 <-> myCobot coords(extrinsic xyz)
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

    HOME_JOINTS = [0, -35, -45, 0, -20, 0]

    def __init__(self, port: str, baud: int = 115200):
        from pymycobot import MyCobot280
        self._mc = MyCobot280(port, baud)
        time.sleep(1)
        self._mc.power_on()
        time.sleep(1)
        self._settle = 2.5

    # ---- 閉環移動 ----
    def move_to_pose(self, pose: np.ndarray, speed: int = 30,
                     gain: float = 0.6,
                     tolerance_mm: float = 8.0,
                     min_improvement_mm: float = 2.0,
                     max_iters: int = 8,
                     verbose: bool = True) -> bool:
        """
        閉環比例控制移動到目標 pose。

        參數:
          gain               每次補償誤差的比例(0.5~0.7 穩;1.0 易震盪)
          tolerance_mm       位置誤差 < 此值即視為到位(預設 8mm,對齊手臂系統誤差級)
          min_improvement_mm 連續改善 < 此值判定「卡住/到極限」而中止(預設 2mm)
          max_iters          最大迭代次數,防呆上限
        回傳:是否收斂到 tolerance 內
        """
        target = np.array(pose_to_coords(pose))

        ok, msg = self._validate(target)
        if not ok:
            raise ValueError(
                f"目標超出 myCobot 280 可命令範圍 -> {msg}\n"
                f"  完整目標: {[round(v,1) for v in target]}\n"
                f"  提示:先 arm.home() 再做相對移動。")

        prev_err = None
        for it in range(1, max_iters + 1):
            current = self._read_coords()
            if current is None:
                if verbose: print(" 讀不到座標,中止")
                return False

            error = target - current              # 位置+姿態誤差向量
            pos_err = float(np.linalg.norm(error[:3]))

            # (2) 收斂檢查
            if pos_err < tolerance_mm:
                if verbose: print(f"  第 {it} 次到位,誤差 {pos_err:.1f}mm")
                return True

            # (3) 卡住檢查:改善太少 = 到極限,別再空轉
            if prev_err is not None and (prev_err - pos_err) < min_improvement_mm:
                if verbose:
                    print(f"  第 {it} 次改善僅 {prev_err - pos_err:.1f}mm(<{min_improvement_mm}),"
                          f"判定到極限。剩餘誤差 {pos_err:.1f}mm")
                return False
            prev_err = pos_err

            # 比例補償:往誤差方向送 gain 倍
            cmd = current + gain * error

            # (1) 範圍守門:每次迭代都檢查,超範圍就 clamp
            cmd = self._clamp(cmd)

            if verbose:
                print(f"  [{it}] 誤差 {pos_err:5.1f}mm -> 送出 "
                      f"[{cmd[0]:.0f},{cmd[1]:.0f},{cmd[2]:.0f}]")
            self._mc.send_coords(cmd.tolist(), speed, 0)   # mode 0 關節插補
            time.sleep(self._settle)

        # 用完迭代次數
        final = self._read_coords()
        pos_err = float(np.linalg.norm((target - final)[:3])) if final is not None else 999
        if verbose:
            print(f"  達最大迭代 {max_iters} 次,剩餘誤差 {pos_err:.1f}mm")
        return pos_err < tolerance_mm

    def home(self, speed: int = 40) -> bool:
        self._mc.send_angles(list(self.HOME_JOINTS), speed)
        time.sleep(5)
        c = self._mc.get_coords()
        if c:
            print(f"home 到位,coords = {[round(v,1) for v in c]}")
        return True

    # ---- 讀取 ----
    def get_tcp_pose(self) -> np.ndarray:
        c = self._read_coords()
        if c is None:
            raise RuntimeError("get_coords 連續讀取失敗")
        return coords_to_pose(c.tolist())

    def _read_coords(self):
        """讀座標,回 np.array(6) 或 None。"""
        for _ in range(3):
            c = self._mc.get_coords()
            if c:
                return np.array(c, dtype=float)
            time.sleep(0.2)
        return None

    # ---- 夾爪 ----
    def grasp(self) -> bool:
        self._mc.set_gripper_state(1, 50); time.sleep(1.5); return True

    def release(self) -> bool:
        self._mc.set_gripper_state(0, 50); time.sleep(1.5); return True

    # ---- 內部:範圍 ----
    @staticmethod
    def _validate(coords):
        for n, v in zip(("x", "y", "z"), coords[:3]):
            if not (XYZ_LIM[0] <= v <= XYZ_LIM[1]):
                return False, f"{n}={v:.1f} 超出 {XYZ_LIM}"
        for n, v in zip(("rx", "ry", "rz"), coords[3:]):
            if not (RPY_LIM[0] <= v <= RPY_LIM[1]):
                return False, f"{n}={v:.1f} 超出 {RPY_LIM}"
        return True, ""

    @staticmethod
    def _clamp(coords):
        """把座標壓回可命令範圍,避免閉環中途送出超範圍指令。"""
        c = np.array(coords, dtype=float)
        c[:3] = np.clip(c[:3], XYZ_LIM[0], XYZ_LIM[1])
        c[3:] = np.clip(c[3:], RPY_LIM[0], RPY_LIM[1])
        return c


# ============================================================
#  示範
# ============================================================
if __name__ == "__main__":
    arm: ArmInterface = MyCobotArm("/dev/ttyAMA0", 1000000)

    print("回工作 home...")
    arm.home()

    pose = arm.get_tcp_pose()
    print("home coords:", [round(v, 1) for v in pose_to_coords(pose)])

    # 抬高 5cm —— 觀察閉環怎麼一步步收斂
    target = pose.copy()
    target[2, 3] += 0.05
    print("\n閉環移動(抬高 50mm):")
    moved = arm.move_to_pose(target, gain=0.6, tolerance_mm=8.0)
    print("最終 Z(mm):", round(arm.get_tcp_pose()[2, 3] * 1000, 1), " 到位:", moved)

    arm.home()