import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:yolo_flutter/bbox.dart';
import 'package:yolo_flutter/labels.dart';
import 'package:yolo_flutter/yolo.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  static const inModelWidth = 640;
  static const inModelHeight = 640;
  static const numClasses = 80;

  static const double maxImageWidgetHeight = 400;

  final YoloModel model = YoloModel(
    'assets/models/yolov8n.tflite',
    inModelWidth,
    inModelHeight,
    numClasses,
  );
  File? imageFile;

  double confidenceThreshold = 0.4;
  double iouThreshold = 0.1;

  List<List<double>>? inferenceOutput;
  List<int> classes = [];
  List<List<double>> bboxes = [];
  List<double> scores = [];

  int? imageWidth;
  int? imageHeight;

  @override
  void initState() {
    super.initState();
    model.init();
  }

  @override
  Widget build(BuildContext context) {
    final bboxesColors = List<Color>.generate(
      numClasses,
      (_) => Color((Random().nextDouble() * 0xFFFFFF).toInt()).withOpacity(1.0),
    );

    final ImagePicker picker = ImagePicker();

    final double displayWidth = MediaQuery.of(context).size.width;

    double resizeFactor = 1;

    if (imageWidth != null && imageHeight != null) {
      double k1 = displayWidth / imageWidth!;
      double k2 = maxImageWidgetHeight / imageHeight!;
      resizeFactor = min(k1, k2);
    }

    List<Bbox> bboxesWidgets = [];
    for (int i = 0; i < bboxes.length; i++) {
      final box = bboxes[i];
      final boxClass = classes[i];
      bboxesWidgets.add(
        Bbox(
            box[0] * resizeFactor,
            box[1] * resizeFactor,
            box[2] * resizeFactor,
            box[3] * resizeFactor,
            labels[boxClass],
            scores[i],
            bboxesColors[boxClass]),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('YOLO')),
      floatingActionButton: FloatingActionButton(
        child: const Icon(Icons.image_outlined),
        onPressed: () async {
          final XFile? newImageFile =
              await picker.pickImage(source: ImageSource.gallery);
          if (newImageFile != null) {
            setState(() {
              imageFile = File(newImageFile.path);
            });
            final image = img.decodeImage(await newImageFile.readAsBytes())!;
            imageWidth = image.width;
            imageHeight = image.height;
            inferenceOutput = model.infer(image);
            updatePostprocess();
          }
        },
      ),
      body: ListView(
        children: [
          SizedBox(
            height: maxImageWidgetHeight,
            child: Center(
              child: Stack(
                children: [
                  if (imageFile != null) Image.file(imageFile!),
                  ...bboxesWidgets,
                ],
              ),
            ),
          ),
          const SizedBox(height: 30),
          const Center(
            child: Text(
              'Confidence threshold',
              style: TextStyle(fontSize: 20),
            ),
          ),
          Slider(
            value: confidenceThreshold,
            min: 0,
            max: 1,
            divisions: 100,
            label: confidenceThreshold.toStringAsFixed(2),
            onChanged: (value) {
              setState(() {
                confidenceThreshold = value;
                updatePostprocess();
              });
            },
          ),
          const SizedBox(height: 30),
          const Center(
            child: Text(
              'IoU threshold',
              style: TextStyle(fontSize: 20),
            ),
          ),
          Slider(
            value: iouThreshold,
            min: 0,
            max: 1,
            divisions: 100,
            label: iouThreshold.toStringAsFixed(2),
            onChanged: (value) {
              setState(() {
                iouThreshold = value;
                updatePostprocess();
              });
            },
          ),
        ],
      ),
    );
  }

  void updatePostprocess() {
    if (inferenceOutput == null) {
      return;
    }
    List<int> newClasses = [];
    List<List<double>> newBboxes = [];
    List<double> newScores = [];
    (newClasses, newBboxes, newScores) = model.postprocess(
      inferenceOutput!,
      imageWidth!,
      imageHeight!,
      confidenceThreshold: confidenceThreshold,
      iouThreshold: iouThreshold,
    );
    debugPrint('Detected ${bboxes.length} bboxes');
    setState(() {
      classes = newClasses;
      bboxes = newBboxes;
      scores = newScores;
    });
  }
}
