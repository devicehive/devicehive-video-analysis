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
python eval.py --video=/path_to_video_file/
```
If _--video_ is not provided video device "0" will be used by default.

Press _q_ to close program.\
Press _s_ to save currents frame to file.

Run 
```bash
python eval.py --help
``` 
for more info about available arguments.

## Supported models

* Yolo9kModel - YOLO model trained to recognize more than 9000 classes.
* Yolo2Model - YOLO-coco model with 80 classes and pretty good accuracy.

Yolo2Model is used by default but this behavior can be changed by passing _--model-name_ argument.
