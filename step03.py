
"""
============================================================
 Step 0.3  兩台 OAK-D Lite 同時取 RGB
============================================================
目標(完成判準):兩台能穩定同時出圖。
這支刻意「最小化」:沒有手臂、沒有存檔、沒有 dataset。
只做一件事 —— 同時開兩台相機,並排顯示兩路即時影像。
出問題才能確定是「相機/USB」的事,不會跟其他東西混在一起。

⚠ Step 0.3 的已知陷阱:雙 OAK 對 USB3 頻寬/供電很敏感。
   若出現掉線 / X_LINK_ERROR / 只認到一台,看檔尾「卡住時怎麼辦」。
============================================================
"""

import cv2
import depthai as dai


def make_pipeline():
    """每台相機各自一份 pipeline(RGB preview 640x640)。"""
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 640)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    # 降頻寬保險:若雙機頻寬吃緊,FPS 調低很有效(預設 30)
    cam.setFps(20)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)
    return pipeline


def main():
    # 1) 先確認到底看得到幾台
    infos = dai.Device.getAllAvailableDevices()
    print(f"偵測到 {len(infos)} 台裝置:")
    for info in infos:
        print(f"   - MxId: {info.getMxId()}  state: {info.state}")

    if len(infos) < 2:
        print("\n❌ 少於兩台。先別急著寫雙機 —— 這通常是 USB 頻寬/供電問題。")
        print("   看檔尾「卡住時怎麼辦」,排除後再跑。")
        return

    # 2) 對每一台,用它的 device_info 各開一個 Device
    devices = []   # (mxid, Device, queue)
    try:
        for info in infos[:2]:
            dev = dai.Device(make_pipeline(), info)
            q = dev.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            devices.append((info.getMxId(), dev, q))
            print(f"✅ 已開啟 {info.getMxId()}  (USB speed: {dev.getUsbSpeed()})")

        print("\n兩台都開了。按 q 離開。")
        print("檢查重點:兩個視窗是否都『持續』更新、不卡頓、不掉線。\n")

        # 3) 主迴圈:兩台都抓最新影格,並排顯示
        while True:
            for mxid, dev, q in devices:
                pkt = q.tryGet()              # 非阻塞,避免一台卡住拖死另一台
                if pkt is not None:
                    frame = pkt.getCvFrame()
                    cv2.imshow(f"cam {mxid[-4:]}", frame)   # 視窗用 id 末四碼區分
            if cv2.waitKey(1) == ord("q"):
                break

    finally:
        # 確實關閉,否則下次可能搶不到裝置
        for _, dev, _ in devices:
            dev.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# ============================================================
#  卡住時怎麼辦(雙 OAK 最常見的就是這幾招)
# ============================================================
"""
症狀:只認到一台 / 跑一下就掉線 / X_LINK_ERROR / 影像卡住

照「最可能 → 最少見」順序試:

1) 供電不足(最常見)
   - OAK-D Lite 從 USB 取電,兩台插同一個被動 hub 幾乎一定供電不夠。
   - 解:用「外部供電(powered)USB hub」,或把兩台插到電腦不同的 USB 孔。

2) USB 頻寬吃緊 / 共用同一個 controller
   - 兩台插在同一排 USB 孔,常常背後共用一個 USB controller,頻寬被分掉。
   - 解:插到電腦「物理上分開」的 USB 孔(例如前面板 + 後面板),
     讓它們落在不同 controller 上。

3) 強制走 USB2(犧牲頻寬換穩定)
   - 若 USB3 一直不穩,可讓相機降速到 USB2,雖然慢但常常就穩了:
        dev = dai.Device(make_pipeline(), info, maxUsbSpeed=dai.UsbSpeed.HIGH)
   - 搭配把 setFps 調更低(例如 10~15)、preview 尺寸調小(例如 416x416)。

4) 線材
   - 換一條「真正支援 USB3 資料傳輸」的短線(很多附贈線只供電/只 USB2)。

驗證口訣:先確認上面 main() 第一段印出「偵測到 2 台」,
        再要求兩個視窗都穩定更新 —— 兩者都達到,Step 0.3 才算過。
"""