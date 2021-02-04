## Bottle Cap Detection Competition WS 20/21
Computer Vision challenge project, [Hochschule Bonn-Rhein-Sieg](https://www.h-brs.de/de), Wintersemester 2020/21. I have been asked to detect and count all bottle caps contained in a at rectangular container from a video, being also able to differentiate them according to the class they belong to.

## Motivation
This project is intended to be an hands on experience with a classic computer vision problem, allowing a better understanding of theoretical ideas behind it. Although this is a toy problem, the techniques developed to approach it can be used, with the necessary modifications, to solve more complex problems.

## Object detection
Object Detection and 3 classes classification problem:
- Bottle Cap Face Up;
- Bottle Cap Face Down;
- Bottle Cap Deformed.

The region of interest (ROI) is a rectangular-shaped with homogenous color,
and it can also contain distractors objects, which we need to be able to distinguish
those from the bottle caps. Also, objects outside the ROI should not be taken into
account.

## General input video sequence 
General Sequence:
- Start from a static (but not necessarily empty) scene;
- Objects are thrown into the ROI;
- The scene to becomes static;
- Objects are removed from the ROI.

Input video             |  Object detection result
:-------------------------:|:-------------------------:
![input video example](input_video_example.gif)  |  <img src="https://github.com/SteEsp/CV-DetectTheBottleCap/blob/main/result_1.png" width="400">

## Tech/framework used
<b>Dependences</b>
- [OpenCV](https://opencv.org/) - Open Source Computer Vision Library.
- [Tensorflow 2](https://www.tensorflow.org/) - The most popular Deep Learning framework created by Google.
- [tqdm](https://github.com/tqdm/tqdm) - Fast, extensible progress bar for loops and CLI.

## Usage example
Launch the program given an input file.
```sh
python main.py -i .\example_1.mp4 -r .\results
python main.py -i .\example_2.mp4 -r .\results
```

## Tests
The main components of the solution's algorithm can be tested separately.

- Extract a static frame (minimum movements)
```sh
cd .\test_modules\test_static_frame_detection\
python main.py -v dataset\videos\CV20_video_1.mp4 -t dataset\images\CV20_image_1.png [-r .\results]
```
- Extraction of the ROI and outsiders objects filtering
```sh
cd .\test_modules\test_roi_detection\
python main.py -f dataset\images\CV20_image_1.png -a dataset\annotations\CV20_label_renamed_1.json [-r .\results]
```
- Object detection predictions
```sh
cd .\test_modules\test_object_detection\
python main.py -f ..\..\dataset\images\CV20_image_1.png -a ..\..\dataset\annotations\CV20_label_renamed_1.json [-r .\results]
```
- Object detection results analysis (notebook)
```sh
notebooks\eval_predictions.ipynb
```

## Credits
Credits to any blogposts or repo that contributed in this project development are cited and linked inside the code.

## Meta

Stefano Esposito – Stefano.Esposito97@outlook.com

[https://github.com/yourname/github-link](https://github.com/SteEsp)
