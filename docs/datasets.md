# Dataset Notes

Raw datasets are not committed in this flagship repo. The repository keeps deployable model artifacts, source code, configuration, and documentation. Full datasets and training runs are kept outside Git history or documented in the module-level repositories.

## Project-Created IMU Driving Events

The IMU driving-event dataset was created for this project and published externally:

- Zenodo: https://zenodo.org/records/20807506
- Kaggle: https://www.kaggle.com/datasets/dinupadevinda/byd-atto-bmi160-imu-driving-events

This dataset is referenced by the IMU modules, including lane-change detection and related driving-event experiments.

## Aggressive Driving

The aggressive-driving module also references selected non-media metadata from:

- Hugging Face: https://huggingface.co/datasets/Stary108/Extreme_Driving_Conditions_Dataset

Only selected metadata-derived features are used in the cleaned module repo. Raw videos, images, depth arrays, LiDAR arrays, and Hugging Face cache files are not committed.

## Road Signs

The road-sign module uses Sri Lankan traffic-sign data from:

- Roboflow Universe: https://universe.roboflow.com/traffic-signs-in-sri-lanka/traffic-signs-in-sri-lanka

The flagship repo only includes the final deploy artifacts used by the Raspberry Pi runtime. Dataset downloads, generated YOLO folders, training runs, and raw images are intentionally excluded.

## Horn Audio

The horn-detection module references this external audio dataset source:

- Mendeley: https://data.mendeley.com/datasets/y5stjsnp8s/2

The cleaned repo keeps the deploy model and runtime documentation, not raw audio files.
