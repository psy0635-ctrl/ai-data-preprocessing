# 가스환경 예측 과제4

Linux 실행 경로: `/workspace/python/Gas_assignment4`

Windows 공유 경로: `D:\dlProject\python\gas_assignment4`

제출 기한: 2026년 11월 10일 오전 9시(KST).

## 결과 확인

- `report.html`: 이미지가 포함된 단일 HTML 보고서. 브라우저에서 열어 확인하거나 인쇄할 수 있습니다.
- `report.md`: 수정하기 쉬운 보고서 원문. 그림은 `figures/`에 있습니다.
- `artifacts/results.json`: 전체 시험 평가 지표.
- `artifacts/test_metrics.csv`: 센서별 원단위 지표.
- `artifacts/model_3s.keras`, `artifacts/model_30s.keras`: 검증 오차 기준으로 선택한 모델.
- `artifacts/training.log`, `artifacts/epochs.jsonl`: 실제 학습 기록.
- `artifacts/independent_audit.json`: 정답 위치·모델 재로딩·평가 지표의 독립 검산 결과.

## 실행 순서

기존 Linux 컨테이너의 `/usr/local/bin/python`을 사용합니다. 패키지를 설치하거나 업그레이드하는 명령은 포함하지 않습니다.

```bash
cd /workspace/python/Gas_assignment4
/usr/local/bin/python inspect_data.py
/usr/local/bin/python prepare_data.py
/usr/local/bin/python experiment.py --verify-only
/usr/local/bin/python experiment.py --epochs 20
/usr/local/bin/python make_report.py
/usr/local/bin/python final_audit.py
/usr/local/bin/python package_submission.py
```

재학습하면 같은 이름의 모델·결과가 갱신됩니다. `epochs.jsonl`은 실행 기록을 추가하므로 여러 번 학습한 경우 실행 구간을 구분해야 합니다.

`assignment4_submission.zip`에는 보고서·그림·코드·모델·평가 결과·검산 기록을 포함하고, 원본 데이터와 재생성 가능한 대용량 `.npy` 캐시는 제외합니다. ZIP 무결성과 포함 파일 SHA-256을 검사합니다.

## 데이터와 설계

`/workspace/DataSet_00000_00999/_Merged_Data`의 CSV 1,000개만 사용합니다. 원본 데이터·Flower 코드를 수정하지 않습니다.

파일 단위 700/150/150 분할, 학습 파일만으로 센서별 Min-Max 계산, index 제외, 센서 6개 입력 및 예측, 과거 100개 관측치, 마지막 입력에서 30행·300행 뒤 정답을 사용합니다. 시간 간격 100ms는 원본 그래프에 근거한 전제입니다.

두 모델은 같은 시작점(0~2600, 5행 간격), 같은 구조와 초기 가중치, 같은 학습 조건을 사용합니다. 시작점 5행 간격은 계산량과 겹치는 윈도우 반복을 줄이기 위한 설계 선택입니다. 각 파일은 521개 샘플을 제공하며 파일 경계를 넘지 않습니다.

모델은 Conv1D 3개와 Dense 2개만 사용합니다. 출력 `(batch,1,6)`의 시간 축 길이 1은 한 미래 시점의 센서 6개를 의미합니다.

센서 6개 동시 예측은 교수님 지침입니다. 각 모델의 마지막 `Dense(6)`가 Temp1, Press1, Temp2, Press2, Acc, Gas를 한 번의 추론으로 함께 출력합니다. 가운데 차원의 1은 센서 개수가 아닌 미래 시점의 개수입니다. 저장 모델의 단일 입력 추론 검산은 `verify_six_outputs.py`와 `artifacts/six_output_verification.json`에 있습니다.

원본 및 변환 캐시는 각각 약 68.7MiB이며, 모든 윈도우를 복사해 저장하지 않습니다. 분할·스케일러·원본 해시는 `artifacts/`에 보존합니다.

실험 해석과 한계는 `report.md`에 설명되어 있습니다.
