# Jetson Orin 실행 가이드

Jetson용 코드는 `src/roshift/speedup_orin/`에 있습니다. Orin용 Python 환경을 별도로 준비해 주세요.

| 실행 대상 | 명령 | 환경 |
|---|---|---|
| RTX 5090 속도 측정 | `roshift speedup ...` 또는 `python -m roshift speedup ...` | Python 3.12, PyTorch 2.10 / CUDA 12.8 |
| 정확도 | `roshift accuracy ...` 또는 `python -m roshift accuracy ...` | 기존 정확도 평가 환경 |
| Jetson Orin 속도 측정 | `roshift speedup-orin ...` 또는 `python -m roshift.speedup_orin ...` | Python 3.10, NVIDIA PyTorch 2.4 / CUDA 12.2 |

## 의존성과 설치

기준 환경은 SM87, Python 3.10.19, PyTorch `2.4.0a0+3bcc3cddb5.nv24.07`, CUDA toolkit 12.2, Triton 3.0.0입니다. PyTorch 배포 메타데이터는 접미사를 `nv24.7`로 표기하지만, 호환성 검사는 런타임 버전인 `nv24.07`을 사용합니다. C++ 컴파일러와 `nvcc`도 필요합니다. 보조 의존성은 [environments/orin.txt](../environments/orin.txt)에 있습니다. 플랫폼 전용 NVIDIA PyTorch와 Triton 빌드는 별도로 설치해 주세요.

SSD에 전용 환경을 만들고 호환되는 NVIDIA PyTorch와 Triton을 준비해 주세요. 데스크톱 환경과 버전 설정은 분리하고, Jetson에 데스크톱용 cu128 wheel을 설치하지 않도록 해 주세요. pip를 실행하기 전에 캐시와 임시 경로를 설정합니다.

```bash
export ROSHIFT_DATA=/ssd/roshift-data
export ORIN_ENV=/ssd/roshift-envs/orin
export TMPDIR="$ROSHIFT_DATA/tmp/speedup_orin"
export PIP_CACHE_DIR="$ROSHIFT_DATA/cache/speedup_orin/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
source "$ORIN_ENV/bin/activate"
export CUDA_HOME=/usr/local/cuda-12.2
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
# Install missing helper packages only inside the dedicated Orin environment.
# python -m pip install -r environments/orin.txt
python -m pip install --no-deps --no-build-isolation .
```

위 설치 명령을 사용하려면 해당 환경에 setuptools >=68이 있어야 합니다.

## 준비된 파일로 실행하기

Orin에서는 `run`, `validate`, `summarize`를 제공하며, `prepare`는 제공하지 않습니다. 필요한 파일은 데스크톱용 준비 절차를 전용 환경에서 실행해 생성한 뒤, 모델별 디렉터리 전체를 Orin에 전달해 주세요.

`--model-path`에는 검증된 체크포인트를, `--artifact-root`에는 호환되는 속도 측정용 전처리 파일을 지정해 주세요. 다음 항목이 필요합니다.

- `weights/s*/manifest.json`과 참조하는 모든 텐서/blob. 상대 경로를 유지해 주세요.
- 대응하는 `centers_s*.json`과 보정/회전 텐서.
- `prompts/prompts.pt`와 데이터셋별 TEAL `thresholds/`.
- 원래의 LM-head 정책과 각 manifest에 필요한 나머지 메타데이터.

Manifest뿐 아니라 참조하는 텐서 파일도 함께 전달해 주세요. 속도 측정용 파일에는 별도의 B128 유지 개수와 측정용 TEAL calibration이 포함되어 있습니다.

Llama-3.1-8B, 희소도 50%의 실행을 확인하시려면 다음 명령을 사용해 주세요.

```bash
python -m roshift.speedup_orin run --model llama31 \
  --model-path /ssd/models/llama31 \
  --artifact-root /ssd/artifacts/llama31 \
  --data-dir "$ROSHIFT_DATA" \
  --output "$ROSHIFT_DATA/speedup_orin/results/llama31_s50_smoke" \
  --percents 50 --datasets wikitext2 --repeats 1 --execute
```

패키지를 설치한 뒤에는 같은 작업을 `roshift speedup-orin`으로 실행할 수 있습니다. `--execute`를 생략하면 GPU 작업을 시작하지 않고 실행할 내용만 보여 줍니다.

Dense, TEAL, RoShift를 FP16, 배치 크기 1, Dense prefill 1024토큰, teacher-forced 컴파일 decode 128 step으로 실행합니다. 기존 워밍업, 측정에서 제외하는 재실행, 출력 검사와 진단을 유지합니다. 측정 중에는 GPU를 다른 프로세스와 공유하지 않도록 해 주세요. 컴파일과 최초 graph capture는 보고하는 decode 지연시간에서 제외합니다.

커널 검증과 결과 요약은 다음과 같이 실행합니다.

```bash
python -m roshift.speedup_orin validate --model llama31 \
  --method roshift --check kernels --percents 50 \
  --data-dir "$ROSHIFT_DATA" --execute
python -m roshift.speedup_orin summarize --model llama31 \
  --data-dir "$ROSHIFT_DATA" \
  --output "$ROSHIFT_DATA/speedup_orin/results/llama31_s50_smoke" --execute
```

커널만 검증할 때는 생성한 텐서를 사용하므로 모델 다운로드가 필요하지 않습니다. 모델 검증에는 체크포인트와 전체 전처리 파일이 필요합니다.

## 저장 경로와 실행 결과

Orin 실행 파일은 `--data-dir` 아래에 저장합니다.

| 용도 | 데이터 디렉터리 아래의 경로 |
|---|---|
| 빌드, 컴파일러, 데이터셋과 드라이버 캐시 | `cache/speedup_orin/` |
| 임시 파일 | `tmp/speedup_orin/` |
| 기본 결과 | `speedup_orin/results/<model>/` |
| 검증된 체크포인트 식별 정보 | `speedup_orin/identity/` |
| 실행 요청 | `requests/speedup_orin/` |

데이터 디렉터리는 SSD에 두고, 별도 `--output`도 SSD 경로를 사용해 주세요. 실행 전 해당 드라이브에 최소 10 GiB가 필요합니다. Orin은 데스크톱과 캐시·결과를 분리하며, 체크포인트와 전처리 파일은 읽기 전용으로 공유할 수 있습니다. 같은 명령을 다시 실행하면 이어서 진행합니다. 코드, 입력이나 설정을 바꾸셨다면 새 출력 디렉터리를 사용해 주세요.
