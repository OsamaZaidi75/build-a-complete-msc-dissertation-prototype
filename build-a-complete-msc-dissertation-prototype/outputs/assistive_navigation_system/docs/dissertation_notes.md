# Dissertation Notes

## Research Aim

Design and prototype an AI-powered assistive navigation system that helps visually impaired users understand nearby obstacles through computer vision, spatial reasoning, simulated haptics, and audio feedback.

## Prototype Contributions

- Real-time YOLO11-based object detection pipeline.
- Spatial field division into left, centre, and right zones.
- Monocular relative distance estimation for obstacle urgency.
- Virtual 270-degree environmental awareness visualization.
- Simulated haptic feedback that maps obstacle direction to vibration intensity.
- Text-to-speech warning generation.
- Streamlit dashboard for experimental observation.
- CSV logging and evaluation scripts for quantitative study.

## Suggested Evaluation Design

- Use annotated videos containing people, chairs, tables, doors, stairs, walls, and vehicles.
- Measure precision and recall using `evaluation/evaluate_detections.py`.
- Measure latency and FPS on CPU and GPU using `evaluation/benchmark.py`.
- Compare warning correctness by zone and warning level.
- Conduct a small user-centered study with simulated haptic/audio output if ethics approval permits.

## Limitations

- Single-camera depth is approximate without calibration.
- COCO-pretrained YOLO11 does not include all dissertation obstacle classes.
- Physical haptic feedback is simulated in software.
- Dashboard latency differs from embedded deployment latency.
