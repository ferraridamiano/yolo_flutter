import 'package:dartcv4/dartcv.dart' as cv;
import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:yolo_flutter/nms.dart';

class YoloModel {
  final String modelPath;
  final int inWidth;
  final int inHeight;
  final int numClasses;
  Interpreter? _interpreter;

  YoloModel(
    this.modelPath,
    this.inWidth,
    this.inHeight,
    this.numClasses,
  );

  Future<void> init() async {
    _interpreter = await Interpreter.fromAsset(modelPath);
  }

  List<List<double>> infer(cv.Mat image) {
    assert(_interpreter != null, 'The model must be initialized');

    final imgResized = cv.resize(image, (inWidth, inHeight));
    final imgRGB = cv.cvtColor(imgResized, cv.COLOR_BGR2RGB);

    // Normalize to 0-1 and convert to Float32
    final imgFloat = imgRGB.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);

    // Get data
    final dataBytes = imgFloat.data;
    final Float32List floatData = Float32List.view(dataBytes.buffer);
    final Uint8List imgUint8 = Uint8List.view(floatData.buffer);

    // output shape:
    // 1 : batch size
    // 4 + 80: left, top, right, bottom and probabilities for each class
    // 8400: num predictions
    final output = [
      List<List<double>>.filled(4 + numClasses, List<double>.filled(8400, 0))
    ];
    int predictionTimeStart = DateTime.now().millisecondsSinceEpoch;
    _interpreter!.run(imgUint8, output);
    debugPrint(
        'Prediction time: ${DateTime.now().millisecondsSinceEpoch - predictionTimeStart} ms');

    // Dispose
    imgResized.dispose();
    imgRGB.dispose();
    imgFloat.dispose();

    return output[0];
  }

  (List<int>, List<List<double>>, List<double>) postprocess(
    List<List<double>> unfilteredBboxes,
    int imageWidth,
    int imageHeight, {
    double confidenceThreshold = 0.7,
    double iouThreshold = 0.1,
    bool agnostic = false,
  }) {
    List<int> classes;
    List<List<double>> bboxes;
    List<double> scores;
    int nmsTimeStart = DateTime.now().millisecondsSinceEpoch;
    (classes, bboxes, scores) = nms(
      unfilteredBboxes,
      confidenceThreshold: confidenceThreshold,
      iouThreshold: iouThreshold,
      agnostic: agnostic,
    );
    debugPrint(
        'NMS time: ${DateTime.now().millisecondsSinceEpoch - nmsTimeStart} ms');
    for (var bbox in bboxes) {
      bbox[0] *= imageWidth;
      bbox[1] *= imageHeight;
      bbox[2] *= imageWidth;
      bbox[3] *= imageHeight;
    }
    return (classes, bboxes, scores);
  }

  (List<int>, List<List<double>>, List<double>) inferAndPostprocess(
    cv.Mat image, {
    double confidenceThreshold = 0.7,
    double iouThreshold = 0.1,
    bool agnostic = false,
  }) =>
      postprocess(
        infer(image),
        image.cols,
        image.rows,
        confidenceThreshold: confidenceThreshold,
        iouThreshold: iouThreshold,
        agnostic: agnostic,
      );
}
