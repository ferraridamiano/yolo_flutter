import 'package:flutter/material.dart';

class Bbox extends StatelessWidget {
  final double x;
  final double y;
  final double width;
  final double height;
  final String label;
  final double score;
  final Color color;

  const Bbox(
    this.x,
    this.y,
    this.width,
    this.height,
    this.label,
    this.score,
    this.color, {
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: y - height / 2,
      left: x - width / 2,
      width: width,
      height: height,
      child: Container(
        decoration: BoxDecoration(
          border: Border.all(color: color, width: 3),
          borderRadius: const BorderRadius.all(Radius.circular(4)),
        ),
        child: Align(
          alignment: Alignment.topLeft,
          child: FittedBox(
            child: Container(
              color: color,
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: <Widget>[
                  Text(label),
                  Text(' ${(score * 100).toStringAsFixed(0)}%'),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
