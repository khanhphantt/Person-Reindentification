<!-- TOC -->

- [Person Re-identification](#person-re-identification-with-openvino)
  - [Intro](#Intro)
  - [Reference](#reference)
    - [OpenVINO Toolkit and Flask Video streaming](#openvino-toolkit-and-flask-video-streaming)
    - [OpenVINO Intel Model](#openvino-intel-model)
  - [Tested Environment](#tested-environment)
  - [Models](#models)
  - [Required Python packages](#required-python-packages)
  - [How to use](#how-to-use)
  - [Run app](#run-app)

<!-- /TOC -->

# Person Re-identification

## Intro

A Person Identification using 2 models from Intel OpenVINO[^1]:

[^1]:See OpenVINO User Guide: [Model Downloader](https://docs.openvino.ai/2022.2/omz_tools_downloader.html)


* Person Detection (person-detection-retail-0013)
* Person Re-Identification (person-reidentification-retail-0288)


## Tested Environment

- Python 3.9.7
- Windows 10
- OpenVINO Toolkit 2022.2


## Required Python packages

```sh
pip install -r requirements.txt
```

## How to use

```sh
python app.py -h
usage: app.py [-h] -i INPUT [-d {CPU,GPU,FPGA,MYRIAD}]
              [-d_reid {CPU,GPU,FPGA,MYRIAD}] [--v4l] [-g GRID] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Paths to video files.
  --m_detector M_DETECTOR
                        Path to the object detection model
  --m_reid M_REID       Path to the object re-identification
                        model
  --config CONFIG       Configuration file
  --loop                Optional. Enable reading the input in
                        a loop
  --t_detector T_DETECTOR
                        Threshold for the object detection model
  -d {CPU,GPU,MYRIAD,HETERO,HDDL}, --device {CPU,GPU,MYRIAD,HETERO,HDDL}
                        Optional. Target device. Default value is CPU.
  --show             Optional. Show output in real time
  --output_video OUTPUT_VIDEO
                        Optional. Path to output video
```


## Run app

**Method 1: Run directly from command line**

```sh
python app.py -i dir_to_video_1 dir_to_video_2 dir_to_video_3
```

**Method 2: Run on browser**

```py
python app.py
```

Access the url below on your browser

```txt
http://127.0.0.1:5000/
```
<img src="https://github.com/khanhphantt/Person-Reindentification/blob/main/demo/demo.gif" alt="mall2" width="%" height="auto">


## Output
**<a href="https://github.com/khanhphantt/Person-Reindentification/blob/main/static/results/ml-program-test-c0%20ml-program-test-c1%20.mp4">
Sample Output Video</a>
