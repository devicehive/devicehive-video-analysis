[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](LICENSE)

# Video analysis demo

## Requirements

### Python
* python 3
* openCV with video support ([Instruction](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html))
* tensorflow ([Instruction](https://www.tensorflow.org/install/install_linux))

### Models data
* download and extract [data.tar.gz](https://s3.amazonaws.com/video-analysis-demo/data.tar.gz) to source folder

## Running

To evaluate video file run
```bash
python eval.py -v /path_to_video_file/
```
If "-v" is not provided video device "0" will be used by default.

Run 
```bash
python eval.py --help
``` 
for more info about available arguments.
