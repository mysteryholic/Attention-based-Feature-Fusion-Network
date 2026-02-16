<img src="docs/open_mmlab.png" align="right" width="22%">

# Attention‑based Feature Fusion Network

이 저장소는 LiDAR 기반 3D 객체 검출 프레임워크인 **OpenPCDet**를 기반으로,
멀티 모달/다중 특징을 **어텐션 기반으로 융합(attention‑based feature fusion)** 하도록
구조를 수정하고, YOLOv5 기반 이미지 모듈을 결합한 연구용 코드베이스입니다.


## 1. 프로젝트 개요

- 베이스 프레임워크: OpenPCDet (LiDAR 3D Object Detection)
- 주요 변경점
	- `pcdet/models`, `pcdet/datasets` 구조를 일부 수정하여
		**attention 기반 feature fusion 모듈**을 추가
	- `tools/` 및 `yolov5/` 디렉터리를 통해 **YOLOv5 2D detector**와 연동 가능
	- custom 데이터셋을 위한 템플릿 및 데이터 전처리 스크립트 제공
- 지원 데이터셋
	- KITTI, NuScenes, Waymo, Lyft, ONCE, Pandaset (OpenPCDet 기본 지원)
	- custom dataset (docs/CUSTOM_DATASET_TUTORIAL.md 참고)

이 저장소는 **OpenPCDet v0.5** 스타일의 학습/평가 파이프라인을
그대로 유지하면서, 추가적인 attention fusion 구조를 실험할 수 있도록 구성되어 있습니다.

---

## 2. 디렉터리 구조

프로젝트 최상단 기준 주요 디렉터리는 다음과 같습니다.

- `pcdet/`
	- 3D detector의 **모델 정의, 데이터셋 정의, 데이터 증강, 후처리** 등이 포함된
		핵심 라이브러리입니다.
	- `datasets/`: KITTI, NuScenes, Waymo, Lyft, ONCE, Pandaset 및 custom 데이터셋 로더
	- `models/`: 3D backbone, neck, head, detector, ROI heads, fusion 모듈 등
	- `ops/`: spconv 기반 연산, ROI pooling, PointNet++ 등 C++/CUDA 확장 모듈
- `tools/`
	- 실험용 스크립트 및 모델 설정 파일이 위치합니다.
	- `train.py`: 단일 GPU 학습 스크립트
	- `test.py`: 학습된 모델 평가 및 결과 저장
	- `demo.py`: 단일 point cloud에 대한 시각화 데모 (docs/DEMO.md 참고)
	- `cfgs/`: 각 데이터셋/모델 조합에 대한 YAML 설정 파일
	- `process_tools/`: 데이터베이스 생성 등 전처리 스크립트
- `yolov5/`
	- YOLOv5 원본 코드가 포함되어 있으며, 2D detector로 사용해
		3D detector와의 **feature/box level fusion** 실험에 활용할 수 있습니다.
- `docs/`
	- `INSTALL.md`: 의존성 설치 및 컴파일 방법
	- `GETTING_STARTED.md`: 데이터셋 준비, 학습/평가 기본 사용법
	- `DEMO.md`: 데모 실행 및 시각화 가이드
	- `CUSTOM_DATASET_TUTORIAL.md`: custom 데이터셋 적용 방법
- `result*.txt`
	- 실험 결과 로그 또는 벤치마크 성능 요약이 텍스트로 기록되어 있습니다.

---

## 3. 설치 (Installation)

자세한 환경 및 설치 방법은 [docs/INSTALL.md](docs/INSTALL.md)를 참고하세요.

### 3.1 기본 요구사항

- OS: Linux / macOS (Linux 기준으로 가장 많이 테스트됨)
- Python: 3.6 이상
- PyTorch: 1.1 이상 (1.3~1.10 권장)
- CUDA: 9.0 이상 (PyTorch 버전에 맞춰 설치)
- spconv: v1.0, v1.2 또는 v2.x 중 택일 (configs와 호환성 확인)

### 3.2 설치 순서 요약

1. 저장소 클론
	 ```bash
	 git clone https://github.com/sanghyunryoo/attention-based-feature-fusion-network.git
	 cd attention-based-feature-fusion-network
	 ```
2. Python 패키지 설치
	 ```bash
	 pip install -r requirements.txt
	 ```
3. spconv 설치
	 - 사용하는 PyTorch/CUDA 버전에 맞는 spconv 버전을 선택하여 설치합니다.
	 - 자세한 내용은 [spconv 공식 문서](https://github.com/traveller59/spconv)를 참고하세요.
4. pcdet 라이브러리 설치
	 ```bash
	 python setup.py develop
	 ```

설치가 끝나면 Python에서 `import pcdet`가 정상적으로 동작해야 합니다.

---

## 4. 데이터셋 준비

데이터셋 준비 절차는 [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)에 상세히 정리되어 있습니다.
여기서는 가장 많이 사용하는 KITTI 기준 예시만 요약합니다.

### 4.1 기본 구조 예시 (KITTI)

```text
attention-based-feature-fusion-network
├── data
│   ├── kitti
│   │   ├── ImageSets
│   │   ├── training
│   │   │   ├── calib, velodyne, label_2, image_2, (optional: planes, depth_2)
│   │   └── testing
│   │       ├── calib, velodyne, image_2
├── pcdet
├── tools
```

### 4.2 데이터 info 생성

KITTI 예시:

```bash
python -m pcdet.datasets.kitti.kitti_dataset \
		create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

NuScenes/Waymo/ONCE/Lyft 등 다른 데이터셋도 비슷한 방식으로
각각의 dataset 스크립트를 호출하여 info 파일을 생성합니다.
필요한 모든 명령어는 GETTING_STARTED.md에 정리되어 있습니다.

custom 데이터셋을 사용하는 경우 [docs/CUSTOM_DATASET_TUTORIAL.md](docs/CUSTOM_DATASET_TUTORIAL.md)를 참고해
annotation 포맷과 폴더 구조를 맞춘 뒤, 대응하는 info 생성 스크립트를 실행하면 됩니다.

---

## 5. 학습 및 평가

### 5.1 단일 GPU 학습

```bash
cd tools
python train.py --cfg_file ${CONFIG_FILE}
```

- `${CONFIG_FILE}` 예시: `cfgs/kitti_models/pv_rcnn.yaml`
- 배치 크기, 에폭 수 등을 커스텀하려면 `--batch_size`, `--epochs` 인자를 추가합니다.

### 5.2 멀티 GPU 학습

```bash
cd tools
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

SLURM 환경에서는 `scripts/slurm_train.sh`를 사용할 수 있습니다.

### 5.3 테스트 및 평가

```bash
cd tools
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

- `--ckpt`에는 학습된 모델의 체크포인트(`.pth`) 경로를 넣습니다.
- `--eval_all` 옵션을 사용하면 실험 디렉터리 내 여러 체크포인트를 일괄 평가할 수 있습니다.
- 멀티 GPU 테스트는 `scripts/dist_test.sh` 또는 SLURM 스크립트를 사용합니다.

---

## 6. 데모 및 시각화

단일 point cloud에 대해 예측 결과를 확인하고 싶다면 [docs/DEMO.md](docs/DEMO.md)를 참고해
아래와 같이 실행할 수 있습니다.

1. Open3D 또는 mayavi 설치
	 ```bash
	 pip install open3d
	 # 또는
	 pip install mayavi
	 ```
2. custom point cloud를 KITTI 기준 좌표계(x: front, y: left, z: up)로 변환
3. `numpy` 포맷 `(N, 4) = [x, y, z, intensity]` 로 저장
4. 데모 실행
	 ```bash
	 cd tools
	 python demo.py \
			 --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
			 --ckpt pv_rcnn_8369.pth \
			 --data_path path/to/your_data.npy
	 ```

실행 후 3D point cloud와 예측 박스가 시각화된 윈도우를 확인할 수 있습니다.

---

## 7. YOLOv5 및 어텐션 기반 융합 구조 개념

이 저장소에는 `yolov5/` 디렉터리가 함께 포함되어 있으며,
2D detector로부터 얻은 정보(예: 2D bounding box, feature map)를
LiDAR 기반 3D detector와 융합하는 구조를 실험할 수 있습니다.

일반적인 흐름은 다음과 같습니다.

1. YOLOv5로 이미지에서 2D detection 수행
2. 각 2D box / feature를 LiDAR 뷰와 정렬 (calibration 사용)
3. `pcdet/models` 내 fusion 모듈에서 **attention 메커니즘**을 이용해
	 - LiDAR feature와 image feature의 중요도를 동적으로 계산
	 - 두 특징을 가중합 또는 concat 후 projection 하는 방식으로 융합
4. 최종 3D detection head에서 3D bounding box 및 class score 예측

구체적인 구현 위치와 사용 방법은 각자의 실험 코드/브랜치에 따라 조금씩 다를 수 있으므로,
`pcdet/models`, `tools/cfgs/custom_models` 등을 참고하여
attention fusion이 포함된 설정 파일을 확인하는 것을 권장합니다.

---

## 8. 자주 보는 파일 정리

- 전체 설정/하이퍼파라미터: `tools/cfgs/**.yaml`
- 학습/평가 스크립트: `tools/train.py`, `tools/test.py`
- 데이터셋 정의: `pcdet/datasets/`
- 모델 및 fusion 모듈: `pcdet/models/`
- 시각화/데모: `tools/demo.py`, `visual_utils/`

---

## 9. 참고

- 본 프로젝트는 OpenPCDet와 YOLOv5의 구조/코드를 기반으로 하며,
	attention 기반 feature fusion 구조를 연구/실험하기 위해 구성되었습니다.
- 원본 프레임워크 사용법이 더 궁금하다면 아래 저장소를 함께 참고하세요.
	- OpenPCDet: https://github.com/open-mmlab/OpenPCDet
	- YOLOv5: https://github.com/ultralytics/yolov5
