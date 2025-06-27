## 🏡 Model Zoo [🔙](../README.md)

### 1. YOLO-MS [🔝](#top)

<table>
  <thead align="center">
    <tr>
      <th>Model</th>
      <th>Epoch</th>
      <th>Params (M)</th>
      <th>FLOPs (G)</th>
      <th>$AP$</th>
      <th>$AP_s$</th>
      <th>$AP_l$</th>
      <th>Config</th>
      <th>🔗</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>YOLO-MS-XS</td>
      <td>300</td>
      <td>5.1M</td>
      <td>8.7G</td>
      <td>42.8</td>
      <td>23.1</td>
      <td>60.1</td>
      <td><a href="../mmyolo/configs/yoloms/yoloms-xs_syncbn_fast_8xb32-300e_coco.py">config</a></td>
      <td>
        <a href="https://drive.google.com/file/d/198s-Iw9eF1am9bTomlRJIxf7FU-n0bWR/view?usp=sharing">model</a> |
        <a href="logs/yoloms/yoloms-xs_syncbn_fast_8xb32-300e_coco.json">Log</a>
      </td>
    </tr>
    <tr>
      <td>YOLO-MS-S</td>
      <td>300</td>
      <td>8.7M</td>
      <td>15.0G</td>
      <td>45.4</td>
      <td>25.9</td>
      <td>62.4</td>
      <td><a href="../mmyolo/configs/yoloms/yoloms-s_syncbn_fast_8xb32-300e_coco.py">config</a></td>
      <td>
        <a href="https://drive.google.com/file/d/1IeWLxH4Tq8xUlAf6byNDFO0zFBkovlWd/view?usp=sharing">model</a> |
        <a href="logs/yoloms/yoloms-s_syncbn_fast_8xb32-300e_coco.json">Log</a>
      </td>
    </tr>
    <tr>
      <td>YOLO-MS</td>
      <td>300</td>
      <td>23.3M</td>
      <td>38.8G</td>
      <td>49.7</td>
      <td>32.8</td>
      <td>65.6</td>
      <td><a href="../mmyolo/configs/yoloms/yoloms_syncbn_fast_8xb32-300e_coco.py">config</a></td>
      <td>
        <a href="https://drive.google.com/file/d/1gX4WxPGVYTM2wdLn49mPzqLSH8lXMv4x/view?usp=sharing">model</a> |
        <a href="logs/yoloms/yoloms_syncbn_fast_8xb32-300e_coco.json">Log</a>
      </td>
    </tr>
  </tbody>
</table>

### 2. YOLO-MS (Previous Version) [🔝](#top)

<table>
  <thead align="center">
    <tr>
      <th>Model</th>
      <th>Epoch</th>
      <th>Params (M)</th>
      <th>FLOPs (G)</th>
      <th>$AP$</th>
      <th>$AP_s$</th>
      <th>$AP_l$</th>
      <th>Config</th>
      <th>🔗</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>YOLO-MS-XS</td>
      <td>300</td>
      <td>4.5</td>
      <td>8.7</td>
      <td>43.1</td>
      <td>24</td>
      <td>59.1</td>
      <td><a href="../mmyolo/configs/yoloms_previous/yoloms-xs_syncbn_fast_8xb32-300e_coco_previous.py">config</a></td>
      <td><a href="https://drive.google.com/file/d/1YDtOcPL4Ot99B1CYv9gwtjspo4TKga0C/view?usp=sharing">model</a></td>
    </tr>
    <tr>
      <td>YOLO-MS-XS*</td>
      <td>300</td>
      <td>4.5</td>
      <td>8.7</td>
      <td>43.4</td>
      <td>23.7</td>
      <td>60.3</td>
      <td><a href="../mmyolo/configs/yoloms_previous/yoloms-xs-se_syncbn_fast_8xb32-300e_coco_previous.py">config</a></td>
      <td><a href="https://drive.google.com/file/d/113tAt8xyzc2ujjN-lc8Aw2hFHDzuNgQ6/view?usp=sharing">model</a></td>
    </tr>
    <tr>
      <td>YOLO-MS-S</td>
      <td>300</td>
      <td>8.1</td>
      <td>15.6</td>
      <td>46.2</td>
      <td>27.5</td>
      <td>62.9</td>
      <td><a href="../mmyolo/configs/yoloms_previous/yoloms-s_syncbn_fast_8xb32-300e_coco_previous.py">config</a></td>
      <td><a href="https://drive.google.com/file/d/1oOQ1eaI50bXuMkFsaN8mznJsiMT-VFvB/view?usp=sharing">model</a></td>
    </tr>
    <tr>
      <td>YOLO-MS-S*</td>
      <td>300</td>
      <td>8.1</td>
      <td>15.6</td>
      <td>46.2</td>
      <td>26.9</td>
      <td>63</td>
      <td><a href="../mmyolo/configs/yoloms_previous/yoloms-s-se_syncbn_fast_8xb32-300e_coco_previous.py">config</a></td>
      <td><a href="https://drive.google.com/file/d/1m4KL7-_N_vo6CHJC7ff_6hdtDWm6jf34/view?usp=sharing">model</a></td>
    </tr>
    <tr>
      <td>YOLO-MS</td>
      <td>300</td>
      <td>22.0</td>
      <td>40.1</td>
      <td>50.8</td>
      <td>33.2</td>
      <td>66.4</td>
      <td><a href="../mmyolo/configs/yoloms_previous/yoloms_syncbn_fast_8xb32-300e_coco_previous.py">config</a></td>
      <td><a href="https://drive.google.com/file/d/1dTSv9ANrgYjCDozwx2ExeDIW9TKHAVPA/view?usp=sharing">model</a></td>
    </tr>
    <tr>
      <td>YOLO-MS*</td>
      <td>300</td>
      <td>22.2</td>
      <td>40.1</td>
      <td>50.8</td>
      <td>33.2</td>
      <td>66.4</td>
      <td><a href="../mmyolo/configs/yoloms_previous/yoloms-se_syncbn_fast_8xb32-300e_coco_previous.py">config</a></td>
      <td><a href="https://drive.google.com/file/d/1YZGflfIq7kcoX-ZQfzMExA9nhdz8AZig/view?usp=sharing">model</a></td>
    </tr>
  </tbody>
</table>

### 3. YOLOv8-MS [🔝](#top)

<table>
  <thead align="center">
    <tr>
      <th>Model</th>
      <th>Epoch</th>
      <th>Params (M)</th>
      <th>FLOPs (G)</th>
      <th>$AP$</th>
      <th>$AP_s$</th>
      <th>$AP_l$</th>
      <th>Config</th>
      <th>🔗</th>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td>YOLOv8-MS-N</td>
      <td>500</td>
      <td>2.9M</td>
      <td>4.4G</td>
      <td>40.2</td>
      <td>20.9</td>
      <td>55.5</td>
      <td><a href="../mmyolo/configs/yolov8_ms/yolov8-ms_n_syncbn_fast_8xb16-500e_coco.py">config</a></td>
      <td>
        <a href="https://drive.google.com/file/d/10YKI2ioJ003m7CxqFIDEH_JbmBxZBPsA/view?usp=sharing">model</a> |
        <a href="logs/yolov8-ms/yolov8-ms_n_syncbn_fast_8xb16-500e_coco.json">Log</a>
      </td>
    </tr>
    <tr>
      <td>YOLOv8-MS-S</td>
      <td>500</td>
      <td>9.5M</td>
      <td>13.3G</td>
      <td>46.2</td>
      <td>27</td>
      <td>62.7</td>
      <td><a href="../mmyolo/configs/yolov8_ms/yolov8-ms_s_syncbn_fast_8xb16-500e_coco.py">config</a></td>
      <td>
        <a href="https://drive.google.com/file/d/15QAACAMIS24DrL-SILE7JOBpM1Xb79lt/view?usp=sharing">model</a> |
        <a href="logs/yolov8-ms/yolov8-ms_s_syncbn_fast_8xb16-500e_coco.json">Log</a>
      </td>
    </tr>
    <tr>
      <td>YOLOv8-MS-M</td>
      <td>500</td>
      <td>25.9M</td>
      <td>35.2G</td>
      <td>Feb-00</td>
      <td>33.6</td>
      <td>65.7</td>
      <td><a href="../mmyolo/configs/yolov8_ms/yolov8-ms_m_syncbn_fast_8xb16-500e_coco.py">config</a></td>
      <td>
        <a href="https://drive.google.com/file/d/1w7ZV6TML9d9s2t7wR2m87-Z8Umb4Q0AQ/view?usp=sharing">model</a> |
        <a href="logs/yolov8-ms/yolov8-ms_m_syncbn_fast_8xb16-500e_coco.json">Log</a>
      </td>
    </tr>
  </tbody>
</table>

### 4. Ablation Study [🔝](#top)

- Ckpts: [https://drive.google.com/drive/folders/1HU9P3P0qj9L-cCo9gVYdpVgkcH2PpYt3?usp=drive_link](https://drive.google.com/drive/folders/1HU9P3P0qj9L-cCo9gVYdpVgkcH2PpYt3?usp=drive_link)
- Logs: [docs/logs/yoloms/ablation](./logs/yoloms/ablation)
