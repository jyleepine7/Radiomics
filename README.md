# NSCLC CT UNet Pipeline

이 프로젝트는 현재 `unbiased CT + segmented mask`만으로 돌아가는 이미지 중심 파이프라인입니다.

지금 구현한 범위:
- 환자별 `image_path`, `mask_path` manifest 읽기
- CT HU windowing, 종양 bounding box crop, axial slice 추출
- 2D UNet segmentation 학습
- 학습된 UNet bottleneck에서 환자별 deep radiomic embedding 추출
- `Radiomics_Clinical.xlsx`에서 clinical/LIFEx sheet 직접 읽기
- 12/36개월 OS/PFS endpoint 생성
- clinical + LIFEx + deep feature를 합친 patient-level table 생성
- nested CV 기반 endpoint 모델링

논문 기반으로 반영한 가정:
- CCI 논문에서 UNet 상세 구조가 없어서, 재현 가능한 표준 2D UNet encoder-decoder로 구현했습니다.
- 현재 deep feature는 UNet bottleneck의 global average pooled slice embedding을 사용하고, 환자 단위에서 `mean + max`로 집계합니다.
- 기본 `base_channels=16` 설정에서는 환자당 deep feature가 512개 생성되어, 논문에서 말하는 deep radiomics 규모를 비슷하게 맞추기 쉽습니다.
- segmented mask가 이미 있으므로, 먼저 segmentation-supervised UNet을 학습하고 그 encoder feature를 downstream radiomics로 사용합니다.
- Clinical/LIFEx 부분은 원 논문의 survival ensemble을 완전히 그대로 복제한 것은 아니고, 현재 버전에서는 nested CV 위에 `logistic regression + random forest + SVM + gradient boosting + MLP` 평균 앙상블로 근사했습니다.

## 폴더 구조

```text
New project/
  config.example.json
  data/
    manifest.example.csv
  scripts/
    run_image_pipeline.py
  src/
    nsclc_unet/
      ...
```

## Manifest 형식

`data/manifest.csv`

```csv
patient_id,image_path,mask_path
L001,/absolute/path/to/L001_ct.nii.gz,/absolute/path/to/L001_mask.nii.gz
L002,/absolute/path/to/L002_ct.nii.gz,/absolute/path/to/L002_mask.nii.gz
```

지원 입력:
- `.nii`
- `.nii.gz`
- `.npy`
- `.npz`
- DICOM directory (`pydicom` 또는 `SimpleITK` 필요)

## 사용법

1. 의존성 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. `config.example.json`을 복사해서 실제 경로로 수정

3. UNet 학습

```bash
python3 scripts/run_image_pipeline.py train --config config.example.json
```

4. 환자별 deep feature 추출

```bash
python3 scripts/run_image_pipeline.py extract --config config.example.json
```

결과물:
- `artifacts/unet_best.pt`
- `artifacts/training_history.json`
- `artifacts/deep_features.csv`

## Colab에서 5명만 빠르게 확인하기

Colab에서는 전체 실험보다 `5명 정도로 코드가 실제로 돌아가고 bottleneck deep feature가 추출되는지`를 먼저 보는 게 안전합니다.

1. 저장소와 의존성 설치

```bash
git clone <YOUR_REPO_URL>
cd "New project"
pip install -r requirements.txt
```

2. `manifest.csv` 준비

```csv
patient_id,image_path,mask_path
L001,/content/data/L001_ct.nii.gz,/content/data/L001_mask.nii.gz
L002,/content/data/L002_ct.nii.gz,/content/data/L002_mask.nii.gz
...
```

3. 5명 smoke test 실행

```bash
python3 scripts/run_colab_smoke_test.py \
  --manifest /content/data/manifest.csv \
  --output-dir /content/artifacts/colab_smoke_test \
  --max-patients 5 \
  --epochs 5 \
  --batch-size 4 \
  --base-channels 8
```

이 스크립트가 하는 일:
- manifest에서 5명만 선택
- `manifest_subset.csv` 자동 생성
- 작은 2D UNet을 짧게 학습
- 환자별 bottleneck deep feature를 `deep_features.csv`로 저장

생성 파일:
- `artifacts/colab_smoke_test/manifest_subset.csv`
- `artifacts/colab_smoke_test/run_summary.json`
- `artifacts/colab_smoke_test/unet_best.pt`
- `artifacts/colab_smoke_test/training_history.json`
- `artifacts/colab_smoke_test/deep_features.csv`

참고:
- 현재 구현은 3D full-volume이 아니라 `tumor-containing axial slice` 기반 2D UNet입니다.
- Colab에서는 먼저 이 smoke test로 파이프라인 검증 후, 이후 GPU 서버에서 더 큰 실험으로 확장하는 흐름을 권장합니다.

## Excel 기반 tabular dataset 만들기

현재 확인한 엑셀 파일:
- `Radiomics_Clinical.xlsx`

patient-level table 생성:

```bash
python3 scripts/run_image_pipeline.py prepare-tabular --config config.example.json
```

이 단계에서 생성되는 것:
- `artifacts/prepared_dataset.csv`

포함 내용:
- clinical 변수
- LIFEx radiomics 집계값
- deep feature CSV가 있으면 자동 병합
- 12/36개월 OS/PFS label과 eligibility flag

## Endpoint 모델 학습

```bash
python3 scripts/run_image_pipeline.py fit-endpoints --config config.example.json
```

결과물:
- `artifacts/endpoint_metrics.json`
- `artifacts/endpoint_predictions.csv`

ROC curve PNG 생성:

```bash
python3 scripts/run_image_pipeline.py plot-roc --config config.example.json
```

결과물:
- `artifacts/roc_curves/roc_label_os_12m.png`
- `artifacts/roc_curves/roc_label_pfs_12m.png`
- 기타 endpoint별 ROC PNG

## 구현 메모

- Clinical 시트는 빈 셀이 많고, Radiomics 시트는 헤더가 한 칸 어긋난 구조가 있어서, `openpyxl` 없이도 읽히도록 XML 기반 `.xlsx` 파서를 직접 넣었습니다.
- Radiomics 시트는 환자당 여러 행이 있을 수 있어, 현재는 numeric scalar feature만 골라 `mean/max/std` 집계가 가능하도록 만들었습니다. 기본값은 `mean + max`입니다.
- OS/PFS horizon label은 censoring을 고려해서, horizon 이전에 censor되면 해당 endpoint 학습에서 제외합니다.
