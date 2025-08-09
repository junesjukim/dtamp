#!/usr/bin/env bash
###############################################################################
# _custom_bash/test_kitchen.sh
#
#  - kitchen-partial-v0 및 kitchen-mixed-v0 환경에 대한 학습을 실행합니다.
#  - 지정된 GPU에 작업을 분산시킵니다.
###############################################################################

######################## 사용자 설정 ##########################################
# 사용할 GPU 장치 목록 (공백으로 구분)
GPU_DEVICES=(2 3)
# 사용할 CPU 스레드 수
OMP_NUM_THREADS=12

# 학습할 D4RL 환경 목록
declare -a ENVS=(
  "kitchen-partial-v0"
  "kitchen-mixed-v0"
)
###############################################################################

# 로그 디렉터리 생성
mkdir -p logs

pids=()
job_idx=0
for ENV in "${ENVS[@]}"; do
    GPU="${GPU_DEVICES[$((job_idx % ${#GPU_DEVICES[@]}))]}"
    LOG_FILE="logs/${ENV}_train.log"
    echo "[실행] GPU ${GPU} | Env ${ENV} | 로그 파일: ${LOG_FILE}"

    CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=${OMP_NUM_THREADS} CUDA_VISIBLE_DEVICES=${GPU} \
    nohup python scripts/d4rl/train_dtamp.py --env "${ENV}" > "${LOG_FILE}" 2>&1 &

    pid=$!
    pids+=("${pid}")
    echo "  [PID: ${pid}] 프로세스 시작됨"
    job_idx=$((job_idx + 1))
done

echo "모든 학습이 실행되었습니다."
