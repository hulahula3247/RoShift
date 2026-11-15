# 정확도 평가 가이드

별도의 Linux/WSL 환경에서 Python 3.12, PyTorch 2.10/cu128을 사용하고, 프로세스마다 CUDA GPU 하나를 할당해 주세요. 정확도 평가는 패키지에 포함된 Transformers/lm-eval 소스를 사용합니다.

Debian/Ubuntu에서는 먼저 `sudo apt install python3.12-venv`를 실행해야 할 수 있습니다.

```bash
export ROSHIFT_DATA=/data/roshift
export PIP_CACHE_DIR="$ROSHIFT_DATA/cache/pip"
export TMPDIR="$ROSHIFT_DATA/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
python3.12 -m venv "$ROSHIFT_DATA/env" &&
source "$ROSHIFT_DATA/env/bin/activate"
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r environments/accuracy.txt
pip install -e .
roshift check
```

Windows에서는 WSL을 사용하고 `ROSHIFT_DATA`, 캐시, 임시 파일과 가상환경을 `/mnt/d/...`에 두세요. C드라이브에는 새 실험 데이터를 만들 수 없습니다. 기존 체크포인트는 어느 디스크에 있든 복사 없이 읽을 수 있습니다. 새 calibration에는 최소 80 GiB, 완료 단계 재사용에는 10 GiB, 선택적 탐색에는 25 GiB의 여유 공간이 필요합니다. 이후 준비 단계에서는 모델 크기에 따라 공간을 확인합니다. GPU마다 모델 하나씩 준비해 주세요.

## 기본 실행: 제공된 최종 설정 사용하기

```bash
export CUDA_VISIBLE_DEVICES=0
roshift accuracy run --model llama31 --method roshift --allocation uniform \
  --sparsity 50 --data-dir "$ROSHIFT_DATA"
roshift accuracy run --model llama31 --method roshift --allocation uniform \
  --sparsity 50 --data-dir "$ROSHIFT_DATA" --execute
```

`--execute`를 생략하면 실행할 내용만 보여 주며, 모델을 불러오거나 다운로드하지 않습니다. 옵션을 추가하면 체크포인트를 다운로드하고 가중치·토크나이저 해시를 확인한 뒤, 통계량/Q/변환된 텐서를 생성하고 선택한 벤치마크를 실행합니다.

접근 승인이 필요한 모델은 먼저 이용 조건에 동의하고 권한을 받아 주세요. 인증에는 로컬 Hugging Face 로그인이나 환경 변수에 저장한 토큰을 사용하고, 소스 파일에 토큰을 직접 적지 않도록 해 주세요. 이미 체크포인트가 있다면 `--model-path /models/existing-checkpoint`로 지정해 읽기 전용으로 재사용하실 수 있습니다.

`--tasks BoolQ PIQA`로 태스크를 선택하거나 `--limit 2`로 짧게 실행하실 수 있습니다. 기본 배치 크기는 1이며, `--batch-size`로 변경합니다. 서버 실험의 설정은 `src/roshift/assets/run_settings.json`에 있습니다. 필요에 따라 `prepare`, `audit`, `ppl`, `evaluate`를 따로 실행하실 수도 있습니다. `audit`는 Dense LM-head를 사용해 calibration 프롬프트 32토큰과 KV 캐시를 사용하는 decode step 1회를 검사합니다.

생성 평가에서는 해당 프로파일의 sparse LM-head calibration과 검증 결과를 확인합니다. PPL 검사는 Mixed calibration, WikiText와 별도로 확보한 Mixed 입력을 사용합니다. Q와 LM-head threshold는 다운스트림 벤치마크의 정답 라벨 없이 선택합니다.

준비 단계에서 calibration, 별도 검증용 입력과 WikiText를 다운로드·복원하고, 기록된 내용 해시를 확인한 뒤 호환되는 완료 단계를 재사용합니다. GPU 없이 입력만 확인하는 명령은 [데이터 이용 안내](DATA_LICENSES.md)를 참고해 주세요.

## 설정을 새로 탐색하려면

```bash
roshift accuracy search --model llama31 --allocation nonuniform --sparsity 50 \
  --search-stage all --data-dir "$ROSHIFT_DATA" --execute
roshift accuracy evaluate --model llama31 --allocation nonuniform --sparsity 50 \
  --use-search --data-dir "$ROSHIFT_DATA" --execute
```

`all`은 TEAL 방식 초기화, WAS 방식 조정, 원래의 1:1 Grid-CD에 이어 Pairwise Greedy와 calibration-PPL 순차 검사를 한 차례 수행합니다. `q`는 고정된 희소도 배분에서 시작하며, `allocation`은 Q 탐색 전에 종료합니다. Uniform 모드는 희소도 배분 탐색을 건너뜁니다. 새 설정은 별도의 `*_research` 작업 디렉터리에 저장합니다.

## 비교 기법

동일한 `accuracy run` 명령에서 `--method dense --sparsity 0`, `--method teal` 또는 `--method larosa`를 지정해 주세요.

- TEAL은 제공된 모델별 공식 histogram과 greedy lookup을 사용합니다. 정확도 벤치마크가 바뀌어도 threshold는 유지합니다. Uniform TEAL은 모든 projection에서 같은 목표 threshold 분위수를 사용하며, RoShift의 희소도 배분기는 사용하지 않습니다.
- LaRoSA는 서버 실험의 WikiText 토큰을 복원하고 해시를 확인한 뒤 모델별 PCA를 다시 생성합니다. Non-uniform 추론에는 제공된 4개 그룹의 MAC 예산 설정을 사용합니다. 추론 회전은 FP16이며, 오프라인 공분산 계산과 고유값 분해는 기록된 정밀도를 유지합니다. 비교 기법의 LM-head는 Dense입니다.

라이선스 정보는 [외부 코드 안내](../third_party/README.md)를 참고해 주세요.

## MBPP 채점

생성 단계에서는 후보 코드 500개를 저장하고 실행하지 않습니다. 모델이나 Hugging Face 인증 정보를 전달하지 않는, root 권한을 사용할 수 있는 별도의 Linux 프로세스에서 채점해 주세요.

```bash
roshift score-mbpp --data-dir /data/scoring --validate-only
roshift score-mbpp --data-dir /data/scoring --input /data/results/mbpp.json
```

채점기는 chroot, UID/GID 권한 하향, seccomp와 자원 제한을 사용하며, 네트워크와 프로세스 생성을 차단합니다. Root 소유의 POSIX 권한이 필요하므로 WSL에서는 메타데이터 기능이 없는 NTFS 대신 ext4 또는 tmpfs 경로를 `--sandbox-root`로 지정해 주세요.

## 결과 저장과 이어서 실행

결과에는 설정, 입력·소스 식별 정보, 점수와 실행시간을 저장합니다. MBPP는 격리된 채점 후 완료됩니다. 같은 명령을 다시 실행하면 식별 정보가 일치하는 완료 결과를 재사용합니다. 입력이나 설정을 바꾸셨다면 새 출력 디렉터리를 사용해 주세요.
