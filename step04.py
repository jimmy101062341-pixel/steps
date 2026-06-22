"""
============================================================
 Step 0.4  ArUco 偵測 hello world(OpenCV 4.7+ objdetect 新 API)
============================================================
完成判準:畫面放一顆 marker,程式印出正確 id 與四角點。

新 API 三步(roadmap 指定,避開舊 aruco.detectMarkers / Dictionary_get):
   dictionary = cv2.aruco.getPredefinedDictionary(...)
   detector   = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
   corners, ids, rejected = detector.detectMarkers(gray)

字典必須與 make_markers.py 一致:DICT_4X4_50。
============================================================
"""

import cv2
import depthai as dai

# ---- 新 API 初始化(全域一次即可)----
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


def make_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 640)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(20)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)
    return pipeline


def main():
    with dai.Device(make_pipeline()) as device:
        q = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        print("開始偵測。把一顆 marker 放到鏡頭前。按 q 離開。\n")

        last_print = None
        while True:
            pkt = q.tryGet()
            if pkt is None:
                if cv2.waitKey(1) == ord("q"):
                    break
                continue

            frame = pkt.getCvFrame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ---- 偵測 ----
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                # 畫出框與 id
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                # 印 id 與四角點(只在內容變動時印,避免洗版)
                summary = []
                for c, i in zip(corners, ids.flatten()):
                    pts = c.reshape(4, 2)
                    summary.append((int(i), pts.round(1).tolist()))
                if summary != last_print:
                    for mid, pts in summary:
                        print(f"id={mid}  四角點(像素)={pts}")
                    last_print = summary

            cv2.imshow("aruco (q to quit)", frame)
            if cv2.waitKey(1) == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# ============================================================
#  偵測不到時的排查順序
# ============================================================
"""
1) 字典不一致 —— 最常見。產生用 DICT_4X4_50,偵測也必須 DICT_4X4_50。
2) 沒留白邊 —— marker 四周要有白色 quiet zone(make_markers.py 已含)。
3) 太遠 / 太糊 / 反光 —— 拉近、對焦、避開反光與陰影。
4) 印歪了 / 貼彎了 —— marker 要平,折到角會偵測不到。
5) 想 debug 可把 rejected candidates 畫出來看它有沒有「快認到」:
       corners, ids, rejected = detector.detectMarkers(gray)
       cv2.aruco.drawDetectedMarkers(frame, rejected, borderColor=(0,0,255))
"""