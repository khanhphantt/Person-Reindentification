"""
app.py: main executation file

__author__      = "Phan Minh Khanh"
__date__        = 16/12/2022
__copyright__   = "Copyright 2022, The Person-Reidentification Project"
__license__     = "Apache"
__version__     = "2.0"
__email__       = "khanhpm@gmail.com"
"""

import configparser
import json
import os
import sys
from logging import (
    DEBUG,
    basicConfig,
    getLogger,
)

import cv2
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
)
from source.args import build_argparser
from source.camera import VideoCamera
from source.interactive_detection import Detections

app = Flask(__name__)
logger = getLogger(__name__)

config = configparser.ConfigParser()
config.read("config.ini")


# detection control flag
is_async = eval(config.get("DEFAULT", "is_async"))
is_det = eval(config.get("DEFAULT", "is_det"))
is_reid = eval(config.get("DEFAULT", "is_reid"))
show_track = eval(config.get("TRACKER", "show_track"))

# 0:x-axis 1:y-axis -1:both axis
flip_code = eval(config.get("DEFAULT", "flip_code"))
resize_width = int(config.get("CAMERA", "resize_width"))


def gen(camera):
    frame_id = 0
    while True:
        frame_id += 1
        frame = camera.get_frame(flip_code)

        if frame is None:
            logger.info("video finished. exit...")
            os._exit(0)
        frame = detections.person_detection(
            frame,
            is_async,
            is_det,
            is_reid,
            str(frame_id),
            show_track,
        )
        ret, jpeg = cv2.imencode(".jpg", frame)
        frame = jpeg.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
        )


@app.route("/")
def index():
    return render_template(
        "index.html",
        is_async=is_async,
        flip_code=flip_code,
        enumerate=enumerate,
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        gen(camera),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/detection", methods=["POST"])
def detection():
    global is_async
    global is_det
    global is_reid
    global show_track

    command = request.json["command"]
    if command == "async":
        is_async = True
    elif command == "sync":
        is_async = False

    if command == "person_det":
        is_det = not is_det
        is_reid = False
    if command == "person_reid":
        is_det = False
        is_reid = not is_reid
    if command == "show_track":
        show_track = not show_track

    result = {
        "command": command,
        "is_async": is_async,
        "flip_code": flip_code,
        "is_det": is_det,
        "is_reid": is_reid,
        "show_track": show_track,
    }
    logger.info(
        f"command:{command} is_async:{is_async} flip_code:{flip_code} is_det:{is_det} is_reid:{is_reid} show_track:{show_track}",  # noqa: E501
    )

    return jsonify(ResultSet=json.dumps(result))


@app.route("/flip", methods=["POST"])
def flip_frame():
    global flip_code

    command = request.json["command"]

    if command == "flip" and flip_code is None:
        flip_code = 0
    elif command == "flip" and flip_code == 0:
        flip_code = 1
    elif command == "flip" and flip_code == 1:
        flip_code = -1
    elif command == "flip" and flip_code == -1:
        flip_code = None

    result = {
        "command": command,
        "is_async": is_async,
        "flip_code": flip_code,
        "is_det": is_det,
        "is_reid": {is_reid},
    }
    return jsonify(ResultSet=json.dumps(result))


if __name__ == "__main__":

    # arg parse
    args = build_argparser().parse_args()
    devices = [args.d_pd, args.d_reid]

    # logging
    # level = INFO
    # if args.verbose:
    level = DEBUG

    basicConfig(
        filename="app.log",
        filemode="w",
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(funcName)s:%(lineno)d %(message)s",  # noqa: E501
    )

    if 0 < args.grid < 3:
        print("\nargument grid must be grater than 3")
        sys.exit(1)

    camera = VideoCamera([args.input, args.input2], resize_width)
    logger.info(
        f"input:{args.input} input2:{args.input2} frame shape: {camera.frame.shape} grid:{args.grid}",  # noqa: E501
    )
    detections = Detections(camera.frame, devices, args.grid)

    app.run(host="0.0.0.0", threaded=True)
