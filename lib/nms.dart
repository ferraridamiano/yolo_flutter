import 'dart:math';

(List<int>, List<List<double>>, List<double>) nms(List<List<double>> rawOutput,
    {double confidenceThreshold = 0.7, double iouThreshold = 0.4}) {
  List<int> bestClasses = [];
  List<double> bestScores = [];

  List<int> boxesToSave = [];

  // Take the argmax to the determine the best classes and scores
  for (int i = 0; i < 8400; i++) {
    double bestScore = 0;
    int bestCls = -1;
    for (int j = 4; j < 84; j++) {
      double clsScore = rawOutput[j][i];
      if (clsScore > bestScore) {
        bestScore = clsScore;
        bestCls = j - 4;
      }
    }
    if (bestScore > confidenceThreshold) {
      bestClasses.add(bestCls);
      bestScores.add(bestScore);
      boxesToSave.add(i);
    }
  }

  // Get rid of boxes below confidence threshold
  List<List<double>> candidateBoxes = [];
  for (var index in boxesToSave) {
    List<double> savedBox = [];
    for (int i = 0; i < 4; i++) {
      savedBox.add(rawOutput[i][index]);
    }
    candidateBoxes.add(savedBox);
  }

  var sortedBestScores = List.from(bestScores);
  sortedBestScores.sort((a, b) => -a.compareTo(b));
  List<int> argSortList =
      sortedBestScores.map((e) => bestScores.indexOf(e)).toList();

  List<int> sortedBestClasses = [];
  List<List<double>> sortedCandidateBoxes = [];
  for (var index in argSortList) {
    sortedBestClasses.add(bestClasses[index]);
    sortedCandidateBoxes.add(candidateBoxes[index]);
  }

  List<List<double>> finalBboxes = [];
  List<double> finalScores = [];
  List<int> finalClasses = [];

  while (sortedCandidateBoxes.isNotEmpty) {
    var bbox1xywh = sortedCandidateBoxes.removeAt(0);
    finalBboxes.add(bbox1xywh);
    var bbox1xyxy = xywh2xyxy(bbox1xywh);
    finalScores.add(sortedBestScores.removeAt(0));
    var class1 = sortedBestClasses.removeAt(0);
    finalClasses.add(class1);

    List<int> indexesToRemove = [];
    for (int i = 0; i < sortedCandidateBoxes.length; i++) {
      if (class1 == sortedBestClasses[i]) {
        if (computeIou(bbox1xyxy, xywh2xyxy(sortedCandidateBoxes[i])) >
            iouThreshold) {
          indexesToRemove.add(i);
        }
      }
    }
    for (var index in indexesToRemove.reversed) {
      sortedCandidateBoxes.removeAt(index);
      sortedBestClasses.removeAt(index);
      sortedBestScores.removeAt(index);
    }
  }
  return (finalClasses, finalBboxes, finalScores);
}

List<double> xywh2xyxy(List<double> bbox) {
  double halfWidth = bbox[2] / 2;
  double halfHeight = bbox[3] / 2;
  return [
    bbox[0] - halfWidth,
    bbox[1] - halfHeight,
    bbox[0] + halfWidth,
    bbox[1] + halfHeight,
  ];
}

/// Computes the intersection over union between two bounding boxes encoded with
/// the xyxy format.
double computeIou(List<double> bbox1, List<double> bbox2) {
  assert(bbox1[0] < bbox1[2]);
  assert(bbox1[1] < bbox1[3]);
  assert(bbox2[0] < bbox2[2]);
  assert(bbox2[1] < bbox2[3]);

  // Determine the coordinate of the intersection rectangle
  double xLeft = max(bbox1[0], bbox2[0]);
  double yTop = max(bbox1[1], bbox2[1]);
  double xRight = min(bbox1[2], bbox2[2]);
  double yBottom = min(bbox1[3], bbox2[3]);

  if (xRight < xLeft || yBottom < yTop) {
    return 0;
  }
  double intersectionArea = (xRight - xLeft) * (yBottom - yTop);
  double bbox1Area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
  double bbox2Area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);

  double iou = intersectionArea / (bbox1Area + bbox2Area - intersectionArea);
  assert(iou >= 0 && iou <= 1);
  return iou;
}
