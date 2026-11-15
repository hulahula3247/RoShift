# CUDA 디코딩 속도 측정 가이드

CUDA 구현으로 Llama-3.1-8B, Llama-2-7B, Mistral-7B-v0.1, Qwen2.5-7B, Qwen3-8B-Base에서 Dense, TEAL, RoShift의 디코딩 지연시간을 측정합니다.

## 실행 환경과 저장 공간

`roshift speedup`은 `src/roshift/speedup/`의 RTX 5090용 구현을 사용합니다. Jetson은 별도 환경과 `roshift speedup-orin`을 사용합니다. [Orin 가이드](ORIN.ko.md)를 참고해 주세요.

Linux/WSL, Python 3.12, PyTorch 2.10.0+cu128, nvcc가 포함된 CUDA 12.8 toolkit, C++ 컴파일러와 `environments/speedup.txt`의 의존성을 준비해 주세요. 준비 단계에는 `environments/accuracy.txt`도 필요합니다. 패키지는 `pip install .`로 설치합니다. `CUDA_VISIBLE_DEVICES`로 GPU 하나를 선택해 주세요. 기준 GPU는 RTX 5090(32 GB)입니다.

체크포인트, 전처리 파일, 캐시와 임시 파일은 `--data-dir` 아래에 저장합니다. Windows C드라이브에는 새 데이터를 만들 수 없습니다. 새 calibration 전에 최소 80 GiB를 확보하고, 모델 하나씩 준비해 주세요. 컴파일과 최초 graph capture는 보고하는 지연시간에서 제외합니다.

## 실행 명령

`--execute`를 생략하면 실행 계획만 확인합니다. 다음 명령은 검증된 원본 체크포인트에서 필요한 calibration/Q/변환된 텐서를 생성하고, 희소 추론 경로의 RMS 중심값과 데이터셋별 TEAL threshold를 구합니다.

```bash
roshift speedup prepare --model llama31 --data-dir /data/roshift --execute
roshift speedup validate --model llama31 --data-dir /data/roshift --execute
roshift speedup run --model llama31 --data-dir /data/roshift --execute
roshift speedup summarize --model llama31 --data-dir /data/roshift --execute
```

체크포인트가 없으면 고정된 리비전을 다운로드하고 해시를 확인합니다. 측정용 프롬프트는 WikiText2, Alpaca, GSM8K, OpenBookQA에서 기록된 Llama 토크나이저를 거쳐 대상 모델의 토크나이저로 다시 처리합니다. 따라서 다른 모델을 실행할 때도 Llama 토크나이저 접근 권한이 있는 HF 계정이 필요합니다.

`--model-path`로 기존 체크포인트를 재사용할 수 있습니다. `--artifact-root`에는 기존 속도 측정용 전처리 파일 디렉터리를 지정합니다. `weights/s*/manifest.json`, `centers_s*.json`, `prompts/prompts.pt`, `thresholds/`가 필요합니다. 파일을 새로 준비할 때는 별도 출력 디렉터리를 사용해 주세요.

작은 범위에서 실행을 확인하시려면 다음 명령을 사용해 주세요.

```bash
roshift speedup run --model llama31 --data-dir /data/roshift \
  --percents 50 --datasets wikitext2 --repeats 1 --execute
```

소스, 프롬프트나 측정 설정을 바꾸셨다면 새로운 `--output`을 사용해 주세요.

## 측정 설정

- FP16, 배치 크기 1, Dense prefill 1024토큰, teacher-forced 희소 decode 128 step, Math SDPA를 사용합니다.
- CUDA event로 decode 시간을 측정하며, torch.compile fullgraph/reduce-overhead/CUDA Graphs를 사용합니다.
- 4개 데이터셋에서 각각 한 행을 사용하고, 데이터셋 순서를 번갈아 가며 4회 반복합니다. 기록하는 실행 전에 워밍업과 측정에서 제외하는 전체 재실행을 진행합니다.
- 기법마다 모델 전체를 한 번 불러온 뒤 같은 프로세스에서 희소도를 25/40/50/60/75% 순서로 높입니다.
- B128 유지 개수는 96/77/64/51/32입니다. 실제 구조적 희소도는 25/39.84375/50/60.15625/75%이며, 정확도 평가의 블록별 분산 배정과는 다릅니다.
- QKV/Gate에는 가중치를 반영한 선택기를 사용하며, permutation seed는 11입니다. O/Up은 원래의 16-bit score-key 선택을 사용하고, Down은 Up에서 선택한 차원을 재사용합니다.
- 모델별 커널 구성은 희소도 50%에서 선택한 설정을 유지합니다. RoShift의 측정 시간에는 비동기 보정, 적응형 대표값 개수, 최종 기저의 가중치 흡수와 3단계 sparse LM-head가 포함됩니다.
- Dense와 TEAL은 Dense LM-head를 사용합니다. TEAL threshold는 측정용 row-0 decode 경로에서 구한 뒤 측정 전에 고정합니다. 정확도 평가는 별도의 공식 threshold 정책을 사용합니다.
- 데이터셋과 반복 실행의 원시 decode 시간(ms)을 평균한 뒤, `tokens/s = 1000 / mean_ms`와 `speedup = paired_dense_ms / sparse_ms`를 계산합니다.

## 전처리

새로 준비할 때는 공개된 **uniform-50 Q 설정**을 측정 희소도 전체에서 사용합니다. 레이어마다 Dense mixed32x4096에서 구한 median 하나를 사용하고, 희소도별 RMS 캐시를 다시 구합니다. 대표값의 총개수는 레이어 수의 4배입니다. 기존 속도 측정 파일은 `--artifact-root`로 재사용하실 수 있습니다.

## 검증

`validate`는 실제 projection 크기에서 무작위 점수, 동점 점수, 0점 입력과 Up 인덱스 순서를 마스크가 적용된 FP32 결과와 비교합니다. 첫 번째 요청 희소도에서는 32토큰 prefill과 KV 캐시를 사용하는 decode step 2회를 실행합니다. Dense는 0%를 사용하며, 커널 검사는 요청한 모든 희소도에서 진행합니다.

`--check kernels` 또는 `--check model`로 검증 범위를 선택합니다. 커널만 검사할 때는 생성한 입력을 사용하므로 모델 가중치를 다운로드하지 않습니다. `run`은 실제 컴파일된 128-step 실행 경로, 대응하는 Dense 실행과의 prefill 일치 여부, 확률, fallback과 LM-head 후보 생존 여부를 측정 구간 밖에서 확인합니다. [검증 명령](VALIDATION.md)도 참고해 주세요.
