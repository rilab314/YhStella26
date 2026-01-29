# SatelliteDet2025

## 1. Execution Guide

### 1.1. Environment Setup

- 저장소 받기: git clone [https://github.com/rilab314/SatelliteDet2025.git](https://github.com/rilab314/SatelliteDet2025.git)
- 폴더 세팅
    - 작업 공간 아래 3개 폴더: 저장소(SatelliteDet2025), dataset, results
- pyenv 설치
- pyenv 환경 생성
- `pip install -r requirements.txt`
- CUDA는 pip으로 가상환경에 설치되므로 시스템에 설치된 CUDA는 쓰지 않는다.
    - .bashrc에서 LD_LIBRARY_PATH 에 CUDA 경로 쓰지 않기

### 1.2. Data Prep

- 실행 스크립트: PROJECT_ROOT/dataset/satellite_lane/generate_label.py
- 경로 설정
    - 프로젝트 폴더 옆에 dataset/seedmap_cfg에 데이터셋 준비
    - 내부 구조 (satellite_lane는 생성될 데이터셋 구조)

```cpp
├── seedmap_cfg
│   ├── image
│   └── label
└── satellite_lane
    ├── test
    │   ├── image
    │   └── label
    ├── train
    │   ├── image
    │   └── label
    └── validation
        ├── image
        └── label
```

- 실행 명령어

```cpp
$ cd $PROJECT_ROOT
$ python dataset/satellite_lane/generate_label.py
```

### 1.3. training

- 실행 명령어: `python pipeline/train.py`
- 결과물 저장: results/tblog_xxxx
    - 모델 체크포인트 저장
    - vlog 폴더에 validation 결과물 저장
- lightning_detr.py에서 vlog_frame_interval 변수로 몇 장마다 visual log를 저장할지 결정 (현재 100장→에폭마다 6장 나옴)
