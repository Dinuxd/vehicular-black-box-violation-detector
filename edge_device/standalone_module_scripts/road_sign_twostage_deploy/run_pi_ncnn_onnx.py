#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import ncnn
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def expanded_square_xyxy(
    xyxy: Sequence[float],
    image_width: int,
    image_height: int,
    margin: float,
) -> Tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = xyxy
    box_w = xmax - xmin
    box_h = ymax - ymin
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    side = max(box_w, box_h) * (1.0 + 2.0 * margin)

    left = center_x - side / 2.0
    top = center_y - side / 2.0
    right = center_x + side / 2.0
    bottom = center_y + side / 2.0

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_width:
        left -= right - image_width
        right = image_width
    if bottom > image_height:
        top -= bottom - image_height
        bottom = image_height

    left = int(max(0, min(round(left), image_width - 1)))
    top = int(max(0, min(round(top), image_height - 1)))
    right = int(max(left + 1, min(round(right), image_width)))
    bottom = int(max(top + 1, min(round(bottom), image_height)))
    return left, top, right, bottom


def softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)


def letterbox_rgb(frame_bgr: np.ndarray, image_size: int) -> Tuple[np.ndarray, float, float, float]:
    height, width = frame_bgr.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))

    resized = cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_w = image_size - resized_width
    pad_h = image_size - resized_height
    left = int(round(pad_w / 2 - 0.1))
    right = int(round(pad_w / 2 + 0.1))
    top = int(round(pad_h / 2 - 0.1))
    bottom = int(round(pad_h / 2 + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB), scale, float(left), float(top)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    xyxy = np.empty_like(boxes, dtype=np.float32)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return xyxy


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        intersection = inter_w * inter_h
        union = areas[current] + areas[rest] - intersection
        ious = intersection / np.maximum(union, 1e-6)
        order = rest[ious <= iou_threshold]

    return keep


class NcnnDetector:
    def __init__(self, model_dir: Path, image_size: int, threads: int) -> None:
        param_path = model_dir / "model.ncnn.param"
        bin_path = model_dir / "model.ncnn.bin"
        if not param_path.exists() or not bin_path.exists():
            raise FileNotFoundError(f"NCNN model files not found in {model_dir}")

        self.net = ncnn.Net()
        self.net.opt.num_threads = max(1, threads)
        if self.net.load_param(str(param_path)) != 0:
            raise RuntimeError(f"Failed to load NCNN param: {param_path}")
        if self.net.load_model(str(bin_path)) != 0:
            raise RuntimeError(f"Failed to load NCNN weights: {bin_path}")
        self.image_size = image_size

    def predict(
        self,
        frame_bgr: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        height, width = frame_bgr.shape[:2]
        image_rgb, scale, pad_left, pad_top = letterbox_rgb(frame_bgr, self.image_size)
        tensor = image_rgb.astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(np.transpose(tensor, (2, 0, 1)))

        extractor = self.net.create_extractor()
        extractor.input("in0", ncnn.Mat(tensor).clone())
        status, output = extractor.extract("out0")
        if status != 0:
            raise RuntimeError("NCNN inference failed while extracting out0")

        raw = np.asarray(output, dtype=np.float32)
        if raw.ndim != 2:
            raise RuntimeError(f"Unexpected NCNN output shape: {raw.shape}")
        detections = raw.T if raw.shape[0] == 5 else raw
        if detections.shape[1] < 5:
            raise RuntimeError(f"Unexpected NCNN detection shape: {detections.shape}")

        scores = detections[:, 4]
        selected = scores >= conf_threshold
        if not np.any(selected):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        boxes = xywh_to_xyxy(detections[selected, :4])
        scores = scores[selected]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)

        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]
        scores = scores[valid]
        keep = nms(boxes, scores, iou_threshold)
        return boxes[keep], scores[keep]


class OnnxCropClassifier:
    def __init__(self, model_path: Path, config: Dict[str, Any], threads: int) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Classifier ONNX not found: {model_path}")

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(1, threads)
        session_options.inter_op_num_threads = 1
        session_options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.classes = list(config["classes"])
        self.input_size = int(config["classifier_input_size"])
        self.mean = np.array(config["imagenet_mean"], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(config["imagenet_std"], dtype=np.float32).reshape(1, 1, 3)

    def predict(self, crop_rgb: np.ndarray) -> Dict[str, Any]:
        resized = cv2.resize(crop_rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - self.mean) / self.std
        tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :]

        logits = self.session.run(None, {self.input_name: tensor})[0]
        probabilities = softmax(np.asarray(logits).reshape(-1))
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
        return {
            "class_index": class_index,
            "class_name": self.classes[class_index],
            "class_conf": confidence,
        }


def draw_label(frame_bgr: np.ndarray, xyxy: Sequence[float], label: str, color: Tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_top = max(0, y1 - text_h - baseline - 6)
    cv2.rectangle(frame_bgr, (x1, label_top), (x1 + text_w + 8, label_top + text_h + baseline + 6), color, -1)
    cv2.putText(
        frame_bgr,
        label,
        (x1 + 4, label_top + text_h + 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_predictions(frame_bgr: np.ndarray, predictions: Sequence[Dict[str, Any]], args: argparse.Namespace) -> None:
    for prediction in predictions:
        xyxy = [
            prediction["xmin"],
            prediction["ymin"],
            prediction["xmax"],
            prediction["ymax"],
        ]
        if prediction["accepted"]:
            label = f"{prediction['class_name']} {prediction['class_conf']:.2f}"
            draw_label(frame_bgr, xyxy, label, (20, 160, 60))
        elif args.draw_rejected:
            label = f"reject {prediction['class_name']} {prediction['class_conf']:.2f}"
            draw_label(frame_bgr, xyxy, label, (120, 120, 120))


def process_frame(
    frame_bgr: np.ndarray,
    detector: NcnnDetector,
    classifier: OnnxCropClassifier,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    boxes, det_scores = detector.predict(frame_bgr, args.det_conf, args.det_iou)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()
    predictions: List[Dict[str, Any]] = []
    reject_class = str(config["reject_class"])

    for det_index, (xyxy, det_score) in enumerate(zip(boxes, det_scores)):
        crop_left, crop_top, crop_right, crop_bottom = expanded_square_xyxy(
            xyxy,
            width,
            height,
            args.crop_margin,
        )
        crop_rgb = frame_rgb[crop_top:crop_bottom, crop_left:crop_right]
        class_result = classifier.predict(crop_rgb)

        reject_reason = ""
        if class_result["class_name"] == reject_class:
            accepted = False
            reject_reason = "reject_class"
        elif class_result["class_conf"] < args.classifier_threshold:
            accepted = False
            reject_reason = "low_confidence"
        else:
            accepted = True

        row = {
            "det_index": det_index,
            "det_conf": float(det_score),
            "xmin": float(xyxy[0]),
            "ymin": float(xyxy[1]),
            "xmax": float(xyxy[2]),
            "ymax": float(xyxy[3]),
            "crop_left": crop_left,
            "crop_top": crop_top,
            "crop_right": crop_right,
            "crop_bottom": crop_bottom,
            "class_name": class_result["class_name"],
            "class_conf": class_result["class_conf"],
            "accepted": accepted,
            "reject_reason": reject_reason,
        }
        predictions.append(row)

        if accepted:
            label = f"{class_result['class_name']} {class_result['class_conf']:.2f}"
            draw_label(annotated, xyxy, label, (20, 160, 60))
        elif args.draw_rejected:
            label = f"reject {class_result['class_name']} {class_result['class_conf']:.2f}"
            draw_label(annotated, xyxy, label, (120, 120, 120))

    return annotated, predictions


def print_terminal_status(
    frame_index: int,
    processed_count: int,
    start_time: float,
    predictions: Sequence[Dict[str, Any]],
) -> None:
    elapsed = max(1e-6, time.time() - start_time)
    accepted = [p for p in predictions if p["accepted"]]
    labels = ", ".join(f"{p['class_name']} cls={p['class_conf']:.2f} det={p['det_conf']:.2f}" for p in accepted)
    if not labels:
        labels = "none"
    print(
        f"frame={frame_index} processed={processed_count} fps={processed_count / elapsed:.2f} "
        f"detected={labels}",
        flush=True,
    )


class AsyncInferenceWorker:
    def __init__(
        self,
        detector: NcnnDetector,
        classifier: OnnxCropClassifier,
        config: Dict[str, Any],
        args: argparse.Namespace,
    ) -> None:
        self.detector = detector
        self.classifier = classifier
        self.config = config
        self.args = args
        self.lock = threading.Lock()
        self.wake_event = threading.Event()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.pending_frame: np.ndarray | None = None
        self.pending_frame_index = 0
        self.latest_predictions: List[Dict[str, Any]] = []
        self.latest_result_frame = 0
        self.rows: List[Dict[str, Any]] = []
        self.processed_count = 0
        self.start_time = time.time()
        self.error: Exception | None = None

    def start(self) -> None:
        self.thread.start()

    def submit(self, frame_index: int, frame_bgr: np.ndarray) -> None:
        with self.lock:
            self.pending_frame = frame_bgr.copy()
            self.pending_frame_index = frame_index
        self.wake_event.set()

    def snapshot(self) -> Tuple[int, List[Dict[str, Any]], int]:
        with self.lock:
            return self.latest_result_frame, list(self.latest_predictions), self.processed_count

    def get_rows(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.rows)

    def raise_if_failed(self) -> None:
        with self.lock:
            error = self.error
        if error is not None:
            raise RuntimeError("Background inference failed") from error

    def stop(self) -> None:
        self.stop_event.set()
        self.wake_event.set()
        self.thread.join()

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            self.wake_event.wait()
            self.wake_event.clear()

            while not self.stop_event.is_set():
                with self.lock:
                    frame_bgr = self.pending_frame
                    frame_index = self.pending_frame_index
                    self.pending_frame = None

                if frame_bgr is None:
                    break

                try:
                    _, predictions = process_frame(
                        frame_bgr,
                        self.detector,
                        self.classifier,
                        self.config,
                        self.args,
                    )
                except Exception as exc:
                    with self.lock:
                        self.error = exc
                    self.stop_event.set()
                    return

                with self.lock:
                    self.latest_predictions = predictions
                    self.latest_result_frame = frame_index
                    self.processed_count += 1
                    processed_count = self.processed_count
                    for prediction in predictions:
                        self.rows.append({"frame": frame_index, **prediction})

                if processed_count % self.args.print_every == 0:
                    print_terminal_status(frame_index, processed_count, self.start_time, predictions)


def inference_process_loop(
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    detector_path: str,
    classifier_path: str,
    config: Dict[str, Any],
    args_dict: Dict[str, Any],
) -> None:
    args = argparse.Namespace(**args_dict)
    try:
        detector = NcnnDetector(Path(detector_path), args.det_imgsz, args.threads)
        classifier = OnnxCropClassifier(Path(classifier_path), config, args.threads)

        processed_count = 0
        while True:
            item = input_queue.get()
            if item is None:
                break

            while True:
                try:
                    newer_item = input_queue.get_nowait()
                except queue.Empty:
                    break
                if newer_item is None:
                    return
                item = newer_item

            frame_index, frame_bgr = item
            _, predictions = process_frame(frame_bgr, detector, classifier, config, args)
            processed_count += 1
            output_queue.put(
                {
                    "ok": True,
                    "frame": frame_index,
                    "predictions": predictions,
                    "processed_count": processed_count,
                }
            )
    except Exception as exc:
        output_queue.put({"ok": False, "error": repr(exc)})


class ProcessInferenceWorker:
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace, classifier_path: Path) -> None:
        self.config = config
        self.args = args
        self.classifier_path = classifier_path
        self.input_queue: mp.Queue = mp.Queue(maxsize=1)
        self.output_queue: mp.Queue = mp.Queue()
        self.process = mp.Process(
            target=inference_process_loop,
            args=(
                self.input_queue,
                self.output_queue,
                str(args.detector),
                str(classifier_path),
                config,
                vars(args).copy(),
            ),
            daemon=True,
        )
        self.latest_predictions: List[Dict[str, Any]] = []
        self.latest_result_frame = 0
        self.rows: List[Dict[str, Any]] = []
        self.processed_count = 0
        self.start_time = time.time()
        self.error: str | None = None

    def start(self) -> None:
        self.process.start()

    def submit(self, frame_index: int, frame_bgr: np.ndarray) -> None:
        self._put_latest((frame_index, frame_bgr.copy()))

    def _put_latest(self, item: Any) -> None:
        while True:
            try:
                self.input_queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    pass

    def snapshot(self) -> Tuple[int, List[Dict[str, Any]], int]:
        self._drain_results()
        return self.latest_result_frame, list(self.latest_predictions), self.processed_count

    def get_rows(self) -> List[Dict[str, Any]]:
        self._drain_results()
        return list(self.rows)

    def raise_if_failed(self) -> None:
        self._drain_results()
        if self.error is not None:
            raise RuntimeError(f"Background inference failed: {self.error}")
        if self.process.exitcode not in (None, 0):
            raise RuntimeError(f"Background inference process exited with code {self.process.exitcode}")

    def stop(self) -> None:
        self._put_latest(None)
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2.0)
        self._drain_results()

    def _drain_results(self) -> None:
        while True:
            try:
                message = self.output_queue.get_nowait()
            except queue.Empty:
                break

            if not message.get("ok"):
                self.error = str(message.get("error", "unknown error"))
                continue

            frame_index = int(message["frame"])
            predictions = list(message["predictions"])
            processed_count = int(message["processed_count"])
            self.latest_result_frame = frame_index
            self.latest_predictions = predictions
            self.processed_count = processed_count
            for prediction in predictions:
                self.rows.append({"frame": frame_index, **prediction})

            if processed_count % self.args.print_every == 0:
                print_terminal_status(frame_index, processed_count, self.start_time, predictions)


def write_predictions_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_video_frame(writer, frame_bgr: np.ndarray, args: argparse.Namespace):
    if not args.save_video:
        return writer

    save_path = Path(args.save_video)
    if not save_path.is_absolute():
        save_path = args.out_dir / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if writer is None:
        height, width = frame_bgr.shape[:2]
        writer = cv2.VideoWriter(
            str(save_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.output_fps,
            (width, height),
        )
    writer.write(frame_bgr)
    return writer


def iter_image_paths(source: Path) -> List[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source]
    if source.is_dir():
        return sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    return []


def run_images(
    source: Path,
    detector: NcnnDetector,
    classifier: OnnxCropClassifier,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    image_paths = iter_image_paths(source)
    if not image_paths:
        raise FileNotFoundError(f"No images found at {source}")

    images_dir = args.out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for image_path in image_paths:
        frame_bgr = cv2.imread(str(image_path))
        if frame_bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        annotated, predictions = process_frame(frame_bgr, detector, classifier, config, args)
        output_path = images_dir / image_path.name
        cv2.imwrite(str(output_path), annotated)
        for prediction in predictions:
            rows.append({"source": str(image_path), "output": str(output_path), **prediction})
        print_terminal_status(0, 1, time.time() - 1.0, predictions)

    write_predictions_csv(args.out_dir / "predictions.csv", rows)
    print(f"Saved outputs to {args.out_dir}")


def open_video_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    source_path = Path(str(args.source))
    source_value: int | str
    if str(args.source).isdigit():
        source_value = int(args.source)
    else:
        source_value = str(args.source)

    capture = cv2.VideoCapture(source_value)
    if str(args.source).isdigit() or is_v4l2_device_path(source_path):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")
    return capture


def is_v4l2_device_path(path: Path) -> bool:
    return path.exists() and path.parent == Path("/dev") and path.name.startswith("video")


def run_video_capture(
    detector: NcnnDetector,
    classifier: OnnxCropClassifier,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    capture = open_video_capture(args)
    writer = None
    rows: List[Dict[str, Any]] = []
    frame_index = 0
    processed_count = 0
    start_time = time.time()

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            frame_index += 1
            if frame_index % args.frame_skip == 0:
                annotated, predictions = process_frame(frame_bgr, detector, classifier, config, args)
                processed_count += 1
                for prediction in predictions:
                    rows.append({"frame": frame_index, **prediction})
                if processed_count % args.print_every == 0:
                    print_terminal_status(frame_index, processed_count, start_time, predictions)
            else:
                annotated = frame_bgr

            writer = maybe_write_video_frame(writer, annotated, args)

            if args.display:
                cv2.imshow("Road sign detection - NCNN + ONNX", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    write_predictions_csv(args.out_dir / "predictions.csv", rows)
    elapsed = max(1e-6, time.time() - start_time)
    print(f"Processed {processed_count} inference frames from {frame_index} camera/video frames.")
    print(f"Inference FPS: {processed_count / elapsed:.2f}")
    print(f"Saved outputs to {args.out_dir}")


def run_video_capture_async(
    config: Dict[str, Any],
    args: argparse.Namespace,
    classifier_path: Path,
) -> None:
    capture = open_video_capture(args)
    worker = ProcessInferenceWorker(config, args, classifier_path)
    worker.start()
    writer = None
    frame_index = 0
    preview_count = 0
    start_time = time.time()

    try:
        while True:
            worker.raise_if_failed()
            ok, frame_bgr = capture.read()
            if not ok:
                break

            frame_index += 1
            preview_count += 1
            if frame_index % args.frame_skip == 0:
                worker.submit(frame_index, frame_bgr)

            result_frame, predictions, _ = worker.snapshot()
            annotated = frame_bgr.copy()
            draw_predictions(annotated, predictions, args)
            if result_frame:
                cv2.putText(
                    annotated,
                    f"latest inference frame {result_frame}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (20, 160, 60),
                    2,
                    cv2.LINE_AA,
                )

            writer = maybe_write_video_frame(writer, annotated, args)

            if args.display:
                cv2.imshow("Road sign detection - async NCNN + ONNX", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        worker.stop()
        capture.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    worker.raise_if_failed()
    rows = worker.get_rows()
    write_predictions_csv(args.out_dir / "predictions.csv", rows)
    _, _, processed_count = worker.snapshot()
    elapsed = max(1e-6, time.time() - start_time)
    print(f"Previewed {preview_count} camera/video frames.")
    print(f"Preview FPS: {preview_count / elapsed:.2f}")
    print(f"Inference FPS: {processed_count / elapsed:.2f}")
    print(f"Saved outputs to {args.out_dir}")


def run_picamera(
    detector: NcnnDetector,
    classifier: OnnxCropClassifier,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    from picamera2 import Picamera2

    camera = Picamera2()
    camera_config = camera.create_preview_configuration(
        main={"format": "RGB888", "size": (args.width, args.height)}
    )
    camera.configure(camera_config)
    camera.start()
    time.sleep(1.0)

    writer = None
    rows: List[Dict[str, Any]] = []
    frame_index = 0
    processed_count = 0
    start_time = time.time()

    try:
        while True:
            frame_rgb = camera.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_index += 1

            if frame_index % args.frame_skip == 0:
                annotated, predictions = process_frame(frame_bgr, detector, classifier, config, args)
                processed_count += 1
                for prediction in predictions:
                    rows.append({"frame": frame_index, **prediction})
                if processed_count % args.print_every == 0:
                    print_terminal_status(frame_index, processed_count, start_time, predictions)
            else:
                annotated = frame_bgr

            writer = maybe_write_video_frame(writer, annotated, args)

            if args.display:
                cv2.imshow("Road sign detection - NCNN + ONNX", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        camera.stop()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    write_predictions_csv(args.out_dir / "predictions.csv", rows)
    elapsed = max(1e-6, time.time() - start_time)
    print(f"Processed {processed_count} inference frames from {frame_index} camera frames.")
    print(f"Inference FPS: {processed_count / elapsed:.2f}")
    print(f"Saved outputs to {args.out_dir}")


def run_picamera_async(
    config: Dict[str, Any],
    args: argparse.Namespace,
    classifier_path: Path,
) -> None:
    from picamera2 import Picamera2

    camera = Picamera2()
    camera_config = camera.create_preview_configuration(
        main={"format": "RGB888", "size": (args.width, args.height)}
    )
    camera.configure(camera_config)
    camera.start()
    time.sleep(1.0)

    worker = ProcessInferenceWorker(config, args, classifier_path)
    worker.start()
    writer = None
    frame_index = 0
    preview_count = 0
    start_time = time.time()

    try:
        while True:
            worker.raise_if_failed()
            frame_rgb = camera.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frame_index += 1
            preview_count += 1

            if frame_index % args.frame_skip == 0:
                worker.submit(frame_index, frame_bgr)

            result_frame, predictions, _ = worker.snapshot()
            annotated = frame_bgr.copy()
            draw_predictions(annotated, predictions, args)
            if result_frame:
                cv2.putText(
                    annotated,
                    f"latest inference frame {result_frame}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (20, 160, 60),
                    2,
                    cv2.LINE_AA,
                )

            writer = maybe_write_video_frame(writer, annotated, args)

            if args.display:
                cv2.imshow("Road sign detection - async NCNN + ONNX", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        worker.stop()
        camera.stop()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    worker.raise_if_failed()
    rows = worker.get_rows()
    write_predictions_csv(args.out_dir / "predictions.csv", rows)
    _, _, processed_count = worker.snapshot()
    elapsed = max(1e-6, time.time() - start_time)
    print(f"Previewed {preview_count} camera frames.")
    print(f"Preview FPS: {preview_count / elapsed:.2f}")
    print(f"Inference FPS: {processed_count / elapsed:.2f}")
    print(f"Saved outputs to {args.out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run exported NCNN detector + ONNX classifier on Raspberry Pi.")
    parser.add_argument("--config", default=str(ROOT / "config.json"), help="Path to config.json.")
    parser.add_argument("--detector", default="models/detector_ncnn/best_ncnn_model", help="NCNN detector folder.")
    parser.add_argument("--classifier", default="", help="Optional classifier ONNX override.")
    parser.add_argument("--source", default="0", help="Image, folder, video path, or USB webcam index. Default: 0.")
    parser.add_argument("--picamera", action="store_true", help="Use Raspberry Pi Camera through Picamera2.")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs_ncnn_onnx"), help="Folder for outputs and CSV.")
    parser.add_argument("--display", action="store_true", help="Show live window. Press q to quit.")
    parser.add_argument("--async-preview", action="store_true", help="Keep camera preview smooth while inference runs in the background.")
    parser.add_argument("--save-video", default="", help="Optional output video filename.")
    parser.add_argument("--output-fps", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frame-skip", type=int, default=2, help="Run inference every N frames.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional limit for testing.")
    parser.add_argument("--threads", type=int, default=4, help="NCNN and ONNX Runtime CPU threads.")
    parser.add_argument("--det-imgsz", type=int, default=0)
    parser.add_argument("--det-conf", type=float, default=-1.0)
    parser.add_argument("--det-iou", type=float, default=-1.0)
    parser.add_argument("--classifier-threshold", type=float, default=-1.0)
    parser.add_argument("--crop-margin", type=float, default=-1.0)
    parser.add_argument("--draw-rejected", action="store_true")
    parser.add_argument("--print-every", type=int, default=1, help="Print terminal status every N processed frames.")
    return parser


def apply_config_defaults(args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    args.config = Path(args.config)
    args.detector = resolve_path(args.detector)
    args.out_dir = Path(args.out_dir)
    args.source_path = Path(args.source)
    args.frame_skip = max(1, args.frame_skip)
    args.print_every = max(1, args.print_every)
    args.det_imgsz = args.det_imgsz or int(config["det_imgsz"])
    args.det_conf = float(config["det_conf"]) if args.det_conf < 0 else args.det_conf
    args.det_iou = float(config["det_iou"]) if args.det_iou < 0 else args.det_iou
    args.classifier_threshold = (
        float(config["classifier_threshold"]) if args.classifier_threshold < 0 else args.classifier_threshold
    )
    args.crop_margin = float(config["crop_margin"]) if args.crop_margin < 0 else args.crop_margin
    return args


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(Path(args.config))
    args = apply_config_defaults(args, config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    classifier_path = resolve_path(args.classifier) if args.classifier else resolve_path(config["classifier_onnx"])
    print(f"Detector NCNN: {args.detector}")
    print(f"Classifier ONNX: {classifier_path}")
    print(
        "Thresholds: "
        f"det_conf={args.det_conf:.2f}, det_iou={args.det_iou:.2f}, "
        f"classifier={args.classifier_threshold:.2f}, crop_margin={args.crop_margin:.2f}"
    )
    if args.async_preview:
        print("Preview mode: async latest-frame inference")

    source_suffix = args.source_path.suffix.lower()
    if args.picamera:
        if args.async_preview:
            run_picamera_async(config, args, classifier_path)
        else:
            detector = NcnnDetector(args.detector, args.det_imgsz, args.threads)
            classifier = OnnxCropClassifier(classifier_path, config, args.threads)
            run_picamera(detector, classifier, config, args)
    elif args.source_path.exists() and (source_suffix in IMAGE_EXTENSIONS or args.source_path.is_dir()):
        detector = NcnnDetector(args.detector, args.det_imgsz, args.threads)
        classifier = OnnxCropClassifier(classifier_path, config, args.threads)
        run_images(args.source_path, detector, classifier, config, args)
    elif str(args.source).isdigit() or is_v4l2_device_path(args.source_path) or source_suffix in VIDEO_EXTENSIONS:
        if args.async_preview:
            run_video_capture_async(config, args, classifier_path)
        else:
            detector = NcnnDetector(args.detector, args.det_imgsz, args.threads)
            classifier = OnnxCropClassifier(classifier_path, config, args.threads)
            run_video_capture(detector, classifier, config, args)
    else:
        raise FileNotFoundError(f"Source not found or unsupported: {args.source}")


if __name__ == "__main__":
    main()
