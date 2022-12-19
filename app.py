"""
app.py: main executation file

__author__      = "Phan Minh Khanh"
__date__        = 16/12/2022
__copyright__   = "Copyright 2022, The Person-Reidentification Project"
__license__     = "Apache"
__version__     = "2.0"
__email__       = "khanhpm@gmail.com"
"""

import logging as log
import os
import queue
import sys
from datetime import date
from logging import getLogger
from threading import Thread

import cv2
from flask import (
    Flask,
    Response,
    flash,
    redirect,
    render_template,
    request,
    send_file,
)
from openvino.runtime import Core
from werkzeug.utils import secure_filename

from sources.args import build_argparser
from sources.mc_tracker.mct import MultiCameraTracker
from sources.settings import (
    DOWNLOAD_FOLDER,
    SECRET_KEY,
    UPLOAD_FOLDER,
)
from sources.thread import FramesThreadBody
from sources.utils.misc import (
    check_pressed_keys,
    read_py_config,
)
from sources.utils.network_wrappers import (
    Detector,
    VectorCNN,
)
from sources.utils.video import (
    MulticamCapture,
    NormalizerCLAHE,
)
from sources.utils.visualization import (
    get_target_size,
    visualize_multicam_detections,
)

# TODO: add to settings
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DOWNLOAD_FOLDER"] = DOWNLOAD_FOLDER
logger = getLogger(__name__)

log.basicConfig(
    format="[ %(levelname)s ] %(message)s",
    level=log.DEBUG,
    stream=sys.stdout,
)


def run(params, config, capture, detector, reid):
    win_name = "person reidentification"
    frame_number = 0
    key = -1

    if config.normalizer_config.enabled:
        capture.add_transform(
            NormalizerCLAHE(
                config.normalizer_config.clip_limit,
                config.normalizer_config.tile_size,
            ),
        )

    tracker = MultiCameraTracker(
        capture.get_num_sources(),
        reid,
        config.sct_config,
        **vars(config.mct_config),
        visual_analyze=config.analyzer,
    )

    global thread_body

    thread_body = FramesThreadBody(
        capture,
        max_queue_length=len(capture.captures) * 2,
    )
    frames_thread = Thread(target=thread_body)
    frames_thread.start()

    prev_frames = thread_body.frames_queue.get()
    detector.run_async(prev_frames, frame_number)
    if len(params.output_video):
        frame_size = [frame.shape[::-1] for frame in prev_frames]
        fps = capture.get_fps()
        target_width, target_height = get_target_size(
            frame_size,
            None,
            **vars(config.visualization_config),
        )

        video_output_size = (target_width, target_height)
        logger.info(f"video_output_size={video_output_size} - {frame_size}")
        output_video = cv2.VideoWriter(
            params.output_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            min(fps),
            video_output_size,
        )
    else:
        output_video = None

    while thread_body.process:
        if params.show:
            key = check_pressed_keys(key)
            if key == 27:
                break
        try:
            frames = thread_body.frames_queue.get_nowait()

        except queue.Empty:
            frames = None

        if frames is None:
            continue

        all_detections = detector.wait_and_grab()
        frame_number += 1
        detector.run_async(frames, frame_number)

        all_masks = [[] for _ in range(len(all_detections))]
        for i, detections in enumerate(all_detections):
            all_detections[i] = [det[0] for det in detections]
            all_masks[i] = [det[2] for det in detections if len(det) == 3]

        tracker.process(prev_frames, all_detections, all_masks)
        tracked_objects = tracker.get_tracked_objects()

        vis = visualize_multicam_detections(
            prev_frames, tracked_objects, **vars(config.visualization_config)
        )

        if params.show:
            cv2.imshow(win_name, vis)

        if output_video:
            output_video.write(cv2.resize(vis, video_output_size))

        prev_frames, frames = frames, prev_frames

        frame = frames[0]
        for id in range(1, len(frames)):
            frame = cv2.hconcat([frame, frames[id]])
        _, jpeg = cv2.imencode(".jpg", frame)
        frame = jpeg.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
        )

    logger.info("Finish process!")
    thread_body.process = False
    frames_thread.join()


def check_file(filename):
    # Validation process
    if filename == "" and filename.endswith((".mp4", ".avi")):
        return True


@app.route("/")
def index():
    return render_template(
        "index.html",
        enumerate=enumerate,
    )


@app.route("/", methods=["POST"])
def upload_video():
    global capture
    global output_name

    if "video_files" not in request.files:
        flash("No file is selected. Please choose 2 videos or more")
        return redirect(request.url)

    uploaded_files = request.files.getlist("video_files")
    prefix = f"{date.today()}"
    original_file = ""
    original_file_wo_ext = ""
    data = []

    if len(uploaded_files) < 2:
        flash("Please choose 2 videos or more")

    for file in uploaded_files:
        if check_file(file.filename) is False:
            flash(
                f"File type is not allowed: {file}. Please select mp4 files.",
            )  # noqa: E501
            return redirect(request.url)

        filename = secure_filename(file.filename)
        original_file += f'"{filename}"' + " "
        original_file_wo_ext += f"{os.path.splitext(filename)[0]}" + " "
        filename = f"{prefix}-{filename}"
        saved_url = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename,
        )
        file.save(saved_url)
        data.append(saved_url)
    flash(f"Analyze {len(uploaded_files)} videos: {original_file}", "video")
    flash(
        "The output video will be available for downloading when the processing is done.",  # noqa: E501
        "download",
    )
    args.input = data
    output_name = original_file_wo_ext + ".mp4"
    args.output_video = os.path.join(
        app.config["DOWNLOAD_FOLDER"],
        output_name,
    )

    capture = MulticamCapture(args.input, args.loop)
    object_detector.max_num_frames = capture.get_num_sources()
    return render_template("index.html", filename=uploaded_files)


@app.route("/process_video")
def process_video():
    return Response(
        run(args, config, capture, object_detector, object_recognizer),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/download_file", methods=["GET"])
def download_file():
    try:
        if thread_body.process is True:
            flash(
                "Processing has not done yet. Please try again later",
                "download",
            )  # noqa: E501
            filename = True
            return render_template("index.html", filename=filename)
        path = os.path.join(
            app.config["DOWNLOAD_FOLDER"],
            output_name,
        )
        # return send_from_directory(directory=path, filename=output_name)
        return send_file(path, as_attachment=True)

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    # arg parse
    args = build_argparser().parse_args()

    if len(args.config):
        log.debug(f"Reading config from {args.config}")
        config = read_py_config(args.config)
    else:
        log.error(
            "No configuration file specified. Please specify parameter '--config'",  # noqa: E501
        )
        sys.exit(1)

    core = Core()
    object_detector = Detector(
        core,
        args.m_detector,
        config.obj_det.trg_classes,
        args.t_detector,
        args.device,
    )

    object_recognizer = VectorCNN(core, args.m_reid, args.device)
    app.run(host="0.0.0.0", threaded=True)
