# NSCLC CT 3D ResNet Pipeline

이 프로젝트는 현재 `unbiased CT + segmented mask`에서 종양 volume을 잘라낸 뒤, `MONAI 3D ResNet`으로 deep feature를 추출하는 이미지 중심 파이프라인입니다.

지금 구현한 범위:
- 환자별 `image_path`, `mask_path` manifest 읽기
- CT HU windowing, 종양 bounding box crop, spacing-aware 3D resample, fixed-size volume resize
- MONAI 3D ResNet backbone checkpoint 준비
- 환자별 penultimate deep embedding 추출
- `Radiomics_Clinical.xlsx`에서 clinical/LIFEx sheet 직접 읽기
- 12/36개월 OS/PFS endpoint 생성
- clinical + LIFEx + deep feature를 합친 patient-level table 생성
- nested CV 기반 endpoint 모델링

현재 가정:
- UNet 재현 대신, `3D cropped tumor volume -> MONAI ResNet embedding` 흐름으로 방향을 바꿨습니다.
- deep feature는 기본적으로 `layer4` 이후 global average pooling 결과를 사용합니다.
- `weights_path`가 없으면 backbone은 랜덤 초기화 상태입니다.
- `weights_path`에 MedicalNet/Med3D 계열 checkpoint를 넣으면 compatible tensor만 backbone에 로드합니다.
- `resnet18` MedicalNet weight를 쓸 때는 `shortcut_type`을 `A`로 맞추는 게 중요합니다.
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
- patient slice directory of `.png` / `.jpg` / `.tif`
- DICOM directory (`pydicom` 또는 `SimpleITK` 필요)

manifest 자동 생성:

```bash
python3 scripts/generate_manifest.py \
  --images-root /path/to/images \
  --masks-root /path/to/masks \
  --output data/manifest.csv \
  --image-suffix _ct \
  --mask-suffix _mask
```

예:
- `/path/to/images/L001_ct.npy` + `/path/to/masks/L001_mask.npy`
- `/path/to/images/L001_ct/` + `/path/to/masks/L001_mask/`

## 사용법

1. 의존성 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. `config.example.json`을 복사해서 실제 경로로 수정

중요:
- `model.weights_path`는 선택 사항입니다.
- Med3D/MedicalNet checkpoint를 직접 구해 놓았으면 그 경로를 넣으세요.
- 비워두면 랜덤 초기화 backbone으로 feature를 추출합니다.
- `resnet18` MedicalNet weight를 연결할 때는 `shortcut_type`을 `A`로 유지하세요.

3. backbone checkpoint 준비

```bash
python3 scripts/run_image_pipeline.py train --config config.example.json
```

4. 환자별 deep feature 추출

```bash
python3 scripts/run_image_pipeline.py extract --config config.example.json
```

결과물:
- `artifacts/resnet18_backbone.pt`
- `artifacts/backbone_summary.json`
- `artifacts/deep_features.csv`

## Colab에서 5명만 빠르게 확인하기

Colab에서는 전체 실험보다 `5명 정도로 코드가 실제로 돌아가고 3D deep feature가 추출되는지`를 먼저 보는 게 안전합니다.

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
  --max-patients 5
```

생성 파일:
- `artifacts/colab_smoke_test/manifest_subset.csv`
- `artifacts/colab_smoke_test/run_summary.json`
- `artifacts/colab_smoke_test/resnet18_backbone.pt`
- `artifacts/colab_smoke_test/deep_features.csv`

## Colab에서 전체 파이프라인 한 번에 돌리기

Drive에 아래 3개가 있다고 가정합니다.
- `/content/drive/MyDrive/Radiomics_colab_data/data/manifest.colab.csv`
- `/content/drive/MyDrive/Radiomics_colab_data/Radiomics_Clinical.xlsx`
- `/content/drive/MyDrive/Radiomics_colab_data/Weights/resnet_18.pth`

실행:

```bash
python3 scripts/run_colab_pipeline.py \
  --manifest /content/drive/MyDrive/Radiomics_colab_data/data/manifest.colab.csv \
  --xlsx-path /content/drive/MyDrive/Radiomics_colab_data/Radiomics_Clinical.xlsx \
  --weights-path /content/drive/MyDrive/Radiomics_colab_data/Weights/resnet_18.pth \
  --output-dir /content/drive/MyDrive/Radiomics_colab_data/output
```

빠른 smoke test로 10명만 먼저 보려면:

```bash
python3 scripts/run_colab_pipeline.py \
  --manifest /content/drive/MyDrive/Radiomics_colab_data/data/manifest.colab.csv \
  --xlsx-path /content/drive/MyDrive/Radiomics_colab_data/Radiomics_Clinical.xlsx \
  --weights-path /content/drive/MyDrive/Radiomics_colab_data/Weights/resnet_18.pth \
  --output-dir /content/drive/MyDrive/Radiomics_colab_data/output_smoke \
  --max-patients 10 \
  --skip-roc
```

이 스크립트가 하는 일:
- generated config 작성
- MedicalNet weight를 MONAI ResNet18 backbone에 로드
- deep feature 추출
- prepared dataset 생성
- endpoint 모델 fitting
- ROC PNG 저장

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

- NIfTI는 내부적으로 `(z, y, x)` 축 순서로 맞춰서 처리합니다.
- DICOM/NIfTI에서 spacing을 읽을 수 있으면 `target_spacing` 기준으로 먼저 resample한 뒤, 최종적으로 `target_depth x target_height x target_width`로 resize합니다.
- Radiomics 시트는 환자당 여러 행이 있을 수 있어, 현재는 numeric scalar feature만 골라 `mean/max/std` 집계가 가능하도록 만들었습니다. 기본값은 `mean + max`입니다.
- OS/PFS horizon label은 censoring을 고려해서, horizon 이전에 censor되면 해당 endpoint 학습에서 제외합니다.
