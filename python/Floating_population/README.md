# 유동인구 시계열 예측

2시간 간격으로 기록된 추정 인구값을 예측하고, 같은 시각과 행렬 위치의 실제값에 맞춰 오차를 계산하는 수업 프로젝트입니다. 요일·시간대 평균, 직전 관측값을 사용하는 Dense, 달력 정보를 보탠 Dense를 비교했습니다.

## 데이터와 환경

수업에서 제공한 ZIP을 풀어 저장소의 `python/Datas_m_3039_H/`에 CSV를 배치합니다. 이 폴더는 `python/Floating_population/`과 같은 수준에 둡니다. 전처리 명령의 `--source-dir`로 다른 위치를 지정할 수 있습니다.

- 예시 파일명은 `20210601_m_3039_H_T_0001.csv`입니다. 날짜와 2시간 구간의 시작 시각을 파일명에서 읽습니다.
- CSV는 UTF-8 BOM, `|` 구분자, 헤더 없는 72행 × 49열 행렬입니다. 배열 축은 `(시간, 행, 열)`이며 행·열은 0부터 셉니다.
- 실행 환경은 Debian 13 기반 Docker 컨테이너 `dlproject`(이미지 `aiswtf`)입니다. 저장소를 `/workspace`에 마운트했습니다. 확인한 Python은 3.12.14이며 NumPy, Matplotlib, TensorFlow 버전은 `requirements.txt`에 있습니다.
- 한글 그래프를 만드는 단계에는 한글 글꼴이 필요합니다. 아래의 `--font`에는 실행 환경에서 읽을 수 있는 글꼴 경로를 지정합니다. 실험에서는 Windows의 맑은 고딕 파일을 컨테이너에 복사해 사용했습니다.

## 실행 순서

컨테이너에 의존성이 없는 경우 `python3 -m pip install -r /workspace/python/Floating_population/requirements.txt`로 설치합니다. 아래 명령은 Windows PowerShell에서 실행합니다. 기본 출력 경로를 사용하는 단계가 있으므로, 기존 결과를 보관하려면 별도 체크아웃에서 재현합니다.

```powershell
docker cp "C:/Windows/Fonts/malgun.ttf" dlproject:/tmp/dlproject-stage2-malgun.ttf

docker exec --workdir /workspace dlproject python3 -B -u python/Floating_population/prepare_data.py --source-dir /workspace/python/Datas_m_3039_H --eval-end 2024-05-31
docker exec --workdir /workspace dlproject python3 -B -u python/Floating_population/explore_train.py --font /tmp/dlproject-stage2-malgun.ttf
docker exec --workdir /workspace -e CUDA_VISIBLE_DEVICES=-1 -e TF_CPP_MIN_LOG_LEVEL=2 dlproject python3 -B -u python/Floating_population/validate_models.py --font /tmp/dlproject-stage2-malgun.ttf

docker exec --workdir /workspace -e CUDA_VISIBLE_DEVICES=-1 -e TF_CPP_MIN_LOG_LEVEL=2 dlproject python3 -B -u python/Floating_population/calendar_dense_experiment.py 2023 forecast
docker exec --workdir /workspace -e CUDA_VISIBLE_DEVICES=-1 -e TF_CPP_MIN_LOG_LEVEL=2 dlproject python3 -B -u python/Floating_population/calendar_dense_experiment.py 2023 score

docker exec --workdir /workspace -e CUDA_VISIBLE_DEVICES=-1 -e TF_CPP_MIN_LOG_LEVEL=2 dlproject python3 -B -W error::UserWarning -u python/Floating_population/evaluate_final.py forecast
docker exec --workdir /workspace -e CUDA_VISIBLE_DEVICES=-1 -e TF_CPP_MIN_LOG_LEVEL=2 dlproject python3 -B -W error::UserWarning -u python/Floating_population/evaluate_final.py score --font /tmp/dlproject-stage2-malgun.ttf

docker exec --workdir /workspace -e CUDA_VISIBLE_DEVICES=-1 -e TF_CPP_MIN_LOG_LEVEL=2 dlproject python3 -B -u python/Floating_population/calendar_dense_experiment.py 2024 forecast
docker exec --workdir /workspace -e CUDA_VISIBLE_DEVICES=-1 -e TF_CPP_MIN_LOG_LEVEL=2 dlproject python3 -B -u python/Floating_population/calendar_dense_experiment.py 2024 score
```

`prepare_data.py`는 관측 배열과 시각 목록을 `prepared/`에 저장합니다. `explore_train.py`는 학습 자료만 사용해 대표 위치, 요일×시간대 히트맵, 계절별 그래프를 `exploration/`에 만듭니다. `validate_models.py`는 두 초기 모델의 내부 검증 결과를 `validation_2023/`에 저장합니다. 최종 평가와 달력 Dense 실험은 예측 파일을 저장한 뒤 별도 `score` 단계에서 실제 인구값을 읽고 오차를 계산합니다.

## 기간과 모델 입력

| 실험 | 적합에 사용한 관측 기간 | 평가에 사용한 관측 기간 | 평가 시각 수 | 비교 셀 수 |
| --- | --- | --- | ---: | ---: |
| 내부 검증 | 2021-06-01~2022-12-31 | 2023-01-01~2023-05-31 | 1,812 | 6,392,736 |
| 최종 테스트 | 2021-06-01~2023-12-31 | 2024-01-01~2024-05-31 | 1,824 | 6,435,072 |

각 기간은 시작일 00:00부터 종료일 22:00에 시작하는 구간까지 포함합니다. 최종 테스트 전 재적합에는 11,328개 시각을 사용했습니다. 전처리 도구가 기본 학습 범위에서 보고하는 2021년 1~5월은 수업 데이터가 제공된 기간 밖입니다.

| 모델 | 입력과 예측 방법 |
| --- | --- |
| 요일·시간대 평균 | 예측 대상과 요일·2시간대가 같은 학습값을 행렬 위치별로 평균 |
| 기존 Dense | 같은 위치의 직전 12개 구간 인구값 |
| 달력 Dense | 직전 인구값 12개와 예측 대상 시각의 2시간대 12칸, 요일 7칸, 월 12칸의 원-핫을 합친 43개 입력 |

두 Dense는 위치별 평균·표준편차를 적합 구간에서 계산합니다. 은닉층은 32·16개 유닛의 tanh, 출력층은 선형이며, Adam(학습률 0.001)으로 MSE를 3에폭 동안 최적화합니다. 시드는 42입니다. 첫 입력 창에는 기준 시각까지의 실제 관측값을 넣고, 이후에는 이전 예측값을 사용해 다음 2시간 구간을 반복 예측합니다. 테스트의 마지막 관측 기준 시각은 2023-12-31 22:00입니다.

## 결과와 한계

MAE와 RMSE는 같은 시각·위치의 셀을 원본 수치 척도에서 비교한 결과입니다. 아래 수치는 재현 시 생성되는 `calendar_dense_experiment/validation_2023/evaluation/metrics.json`과 `calendar_dense_experiment/test_2024/evaluation/metrics.json`의 `overall` 항목에서 읽어 소수점 아래 여섯 자리로 표시했습니다.

| 모델 | 2023년 내부 검증 MAE | RMSE | 2024년 1~5월 테스트 MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| 요일·시간대 평균 | 2.115792 | 13.930321 | 8.495874 | 55.917191 |
| 기존 Dense | 4.486317 | 31.102525 | 23.174868 | 156.173600 |
| 달력 Dense | 3.106328 | 19.698999 | 9.604535 | 60.092135 |

처음 두 모델의 내부 검증 결과를 바탕으로 요일·시간대 평균을 주 모델로 선정했습니다. 달력 Dense는 과제 지침을 반영한 후속 실험입니다. 앞선 두 모델의 테스트 결과는 이미 확인한 상태였으며, 달력 Dense 설정은 2021~2023년 자료를 사용해 정했습니다. 달력 Dense는 기존 Dense보다 오차가 낮았고, 2024년 테스트에서는 요일·시간대 평균의 오차가 가장 낮았습니다.

달력 Dense는 입력과 매개변수 수가 함께 바뀌어서 시간대·요일·월 개별 효과를 구분하기 어렵습니다. 반복 예측에서는 앞 구간의 오차가 다음 입력에 전달될 수 있습니다. 두 Dense의 선형 출력에서 나온 음수도 원본 척도로 되돌린 뒤 채점에 포함했습니다. 행렬 위치의 지리 좌표와 0이 기록된 이유는 제공 자료에서 확인할 수 없었습니다.
