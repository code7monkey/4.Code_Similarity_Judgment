# GraphCodeBERT 코드 유사도 분석 프로젝트

이 저장소는 GraphCodeBERT 모델을 활용하여 코드 쌍의 유사도를 판별하는 분류 모델을 학습하고 추론하는 예시 프로젝트입니다.  
원본 Jupyter 노트북에서 작성된 실험 코드를 모듈화된 Python 패키지로 정리하여 재사용성과 가독성을 높였습니다.

## 🔥 프로젝트 목표

* **코드 전처리 및 페어 생성** – 소스 코드에서 주석과 불필요한 공백을 제거하는 `remove_extras` 함수를 제공하고, 훈련을 위해 코드 긍정/부정 쌍을 생성합니다.
* **GraphCodeBERT 분류기** – HuggingFace의 `AutoModelForSequenceClassification`을 래핑하여 코드 유사도를 예측하는 모델을 구현합니다.
* **모듈화된 학습/추론 파이프라인** – `src/trainer.py`에 학습 루프와 추론 루틴을 정의하고, 상위 스크립트에서 YAML 설정 파일을 읽어 실행합니다.
* **손쉬운 실험 조정** – `configs/train.yaml`과 `configs/submit.yaml`을 통해 데이터 경로, 하이퍼파라미터, 모델 저장 경로 등을 손쉽게 수정할 수 있습니다.

## 📂 디렉터리 구조

```
graphcodebert_project/
├── src/
│   ├── __init__.py           # 패키지 초기화
│   ├── utils.py              # 시드 고정 및 코드 전처리 함수
│   ├── dataset.py            # CodePairDataset 및 데이터셋 생성 헬퍼
│   ├── model.py              # GraphCodeBERT 분류기 래퍼
│   └── trainer.py            # 학습, 검증, 추론 루틴
│
├── train.py                  # 학습 실행 스크립트
├── inference.py              # 추론 실행 스크립트
├── configs/
│   ├── train.yaml            # 학습 설정 파일
│   └── submit.yaml           # 추론 설정 파일
├── requirements.txt          # 필요한 파이썬 패키지 목록
├── assets/
│   ├── model1.pt             # 첫 번째 모델 가중치(예시)
│   └── model2.pt             # 두 번째 모델 가중치(예시)
├── data/
│   ├── train.csv             # 학습 데이터셋(예시)
│   ├── test.csv              # 테스트 데이터셋(예시)
│   └── sample_submission.csv # 제출 형식 예시
├── .gitignore                # Git이 무시할 파일 목록
└── .gitattributes            # Git 속성 정의 (예: LFS 설정)
```

## 🚀 시작하기

### 환경 준비

Python 3.9 이상 환경을 권장합니다. 프로젝트 루트에서 다음 명령어를 실행해 필요한 라이브러리를 설치하세요.

```bash
pip install -r requirements.txt
```

### 학습 실행

학습 데이터를 준비한 후 아래 명령어로 모델을 학습할 수 있습니다. 설정 값은 `configs/train.yaml`에서 수정 가능합니다.

```bash
python train.py --config configs/train.yaml
```

### 추론 실행

훈련된 모델 가중치를 로드하여 테스트 세트에 대한 예측을 생성하려면 다음을 실행합니다.

```bash
python inference.py --config configs/submit.yaml
```

추론 스크립트는 하나 이상의 모델 가중치를 로드하여 예측을 앙상블하고, `sample_submission.csv` 포맷에 맞는 결과를 생성합니다.

## 참고 사항

* `assets/model1.pt`와 `assets/model2.pt`는 예시용 빈 파일입니다. 실제 사용 시 학습된 모델 가중치를 이 위치에 저장하세요.
* `data/train.csv`와 `data/test.csv`에는 간단한 코드 쌍과 라벨이 포함돼 있습니다. 실제 프로젝트에서는 해당 부분을 자신의 데이터로 교체해야 합니다.
* 학습/추론 파이프라인은 기본적인 구현입니다. 필요에 따라 손실 함수, 평가 지표, 데이터 로더 등을 확장해 사용하세요.
