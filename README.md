[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](LICENSE)

# Video analysis demo

## Requirements

### Python
* openCV with video support ([Instruction](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html))
* tensorflow ([Instruction](https://www.tensorflow.org/install/install_linux))
* [devicehive-python-webconfig](https://github.com/devicehive/devicehive-python-webconfig) (for daemon.py)
* pafy (for direct video links)
* youtube-dl (for direct video links from [youtube](https://www.youtube.com))

All requirements are listed in [requirements](requirements.txt) file, ~~except _openCV_ (because of video support described [here](https://pypi.python.org/pypi/opencv-python#frequently-asked-questions), [here](https://github.com/skvark/opencv-python/issues/49) and [here](https://github.com/skvark/opencv-python/issues/50))~~

So ~~(almost)~~ all required packages can be installed by _pip_

**Update:** Since [this](https://github.com/skvark/opencv-python/releases/tag/11) release opencv-python ships with FFmpeg, V4L and V4L2. So i have added it to [requirements](requirements.txt).
```bash
pip install -r requirements.txt
```

### Models data
* download and extract [data.tar.gz](https://s3.amazonaws.com/video-analysis-demo/data.tar.gz) to source folder

## Running

### Local demo
To evaluate video file run
```bash
python eval.py --video="/path_to_video_file/"
```

Also video url can be passed (tested for youtube)
```bash
python eval.py --video="youtube_url"
```

If _--video_ is not provided video device "0" (usually it's web cam) will be used by default.

Press _q_ to close program.\
Press _s_ to save currents frame to file.

Run 
```bash
python eval.py --help
``` 
for more info about available arguments.

### Web based demo
There is demo with web interface and devicehive integration.
It will capture video stream from your web cam, evaluate it and send predictions to devicehive.
To use it run
```bash
python daemon.py
```
and go to http://127.0.0.1:8000 to configure devicehive connection.
Video stream is available on http://127.0.0.1:8000/events/
