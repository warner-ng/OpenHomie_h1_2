import pyrealsense2 as rs
import numpy as np
import cv2

import socket
import struct
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

d455_pipeline = rs.pipeline()
d455_pipeline.start(config)
ip = "192.168.1.203"
port = 9998
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((ip, port))

try:
    while True:
        frames = d455_pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, img_encoded = cv2.imencode('.jpg', color_image, encode_param)
        data = img_encoded.tobytes()
        data_size = len(data)
        client_socket.sendall(struct.pack(">I", data_size) + data)

finally:
    d455_pipeline.stop()
    cv2.destoryAllWindows()