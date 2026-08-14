# 실험 19/24/25는 원본 데이터 자체의 중복(사후 발견, 아래 참고)이라 제외한다:
# - 실험14(train)와 실험24(eval)는 센서 데이터가 완전히 동일함(md5 일치) — 같은 실험이
#   train과 eval에 동시에 들어가는 리키지라 24를 제외.
# - 실험19(불량)와 실험25(양품)는 센서 데이터가 완전히 동일한데 라벨(및 feedrate/
#   clamp_pressure)이 서로 모순됨 — 어느 쪽 라벨도 신뢰할 수 없어 둘 다 제외.
EXCLUDED_DUPLICATE_EXPERIMENT_IDS = [19, 24, 25]

TRAIN_EXPERIMENT_IDS = [1, 2, 3, 11, 13, 14, 15, 17]
EVAL_GOOD_EXPERIMENT_IDS = [12, 18, 22]
EVAL_BAD_EXPERIMENT_IDS = [4, 5, 6, 7, 8, 9, 10, 16, 20, 21, 23]
