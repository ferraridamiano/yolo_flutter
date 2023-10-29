# yolo_flutter

An example project using the official [`tflite_flutter`](https://pub.dev/packages/tflite_flutter)
package from the tensorflow team to run YOLO (a fast object detection model).
This example is complete: it embeds the non-max suppression algorithm I wrote in
Dart. My implementation is not optimized and it takes ~70 ms on a Snapdragon 730
just for this post-processing step. The app should be compatible with iOS, MacOS
and Android (although I only tested the latter).

<div align="center">
    <img src='images/image_1.jpg' width='200'>
    <img src='images/image_2.jpg' width='200'>
</div>

## How to compile it
- [Install flutter](https://docs.flutter.dev/get-started/install)
- [Install the ultralytics Python package](https://docs.ultralytics.com/quickstart)
- Export the yolo model to tflite, e.g. `yolo export model=yolov8n.pt format=tflite`
- Place the output model in the `assets/models` folder as `yolov8n.tflite`
- Compile the app with `flutter build apk --release`

## FAQ

### Can I run my custom object detection model?
Yes, place your model in the assets/models folder and change the labels inside
the `lib/labels.dart` file.

### Does it work with all the versions of YOLO?
I tested it only with YOLOv8 but you could try and let me know. The most 
important thing is to make sure that the output format of the neural network
is the same as YOLOv8.

