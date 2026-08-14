# 폴더 구조 정리 (2026-08-14)

## 목적

멘토·팀원이 저장소를 처음 클론했을 때, 폴더 목록만 보고 **무엇을 먼저
봐야 하는지** 바로 알 수 있게 만든다. 현재는 루트에 README가 없고
`version_2/`라는 이름이 트랙의 정체를 감추고 있어, 처음 보는 사람이
접힌 1차 트랙(CN7)을 메인으로 오해하기 쉽다.

## 결정 사항

- 두 트랙에 번호 접두사를 붙여 진행 순서를 드러낸다
  (`01-cn7-injection-molding/`, `02-cnc-machining/`)
- CNC 트랙 내부의 실험 폴더(`loocv/`, `synthetic/`, `rag/`,
  `augmentation/`, `monitoring/`)는 **건드리지 않는다** — 스크립트가
  `Path(__file__).parent.parent`로 루트를 역산해서 깊이가 코드에 박혀
  있고, 통합 이득보다 회귀 위험이 크다. README 설명으로 해결한다.
- `pyproject.toml`의 `name` 필드는 **바꾸지 않는다** — uv_build의 패키지
  탐색 키라서 변경 시 `module-name` 오버라이드가 추가로 필요하다.
  `description`만 정확하게 손본다.

## 작업

- [x] 1. `01-cn7-injection-molding/` 생성 후 루트의 CN7 트랙 이동
      → 검증: `git status`에 `R`(rename)로 잡히는지
- [x] 2. `version_2/` → `02-cnc-machining/` rename
      → 검증: 스크립트 `ROOT` 계산 깊이 불변 확인
- [x] 3. `docs/superpowers/{specs,plans}` → `docs/{specs,plans}` 평탄화,
      결과 리포트 docx를 `docs/`로 이동
      → 검증: 두 트랙의 문서 경로 규칙이 같은지
- [x] 4. README 3종 (루트 신규 / 01번 신규 / 02번 경로 문구 갱신)
      → 검증: `grep -rn version_2` 잔여 0건
- [x] 5. 전체 검증 (테스트 통과, 잔여물 정리)
      → 검증: 두 트랙 `uv run pytest` 통과

## 리뷰

### 결과

```
hanium/
├── README.md                      신규 — 진입점, 두 트랙 서사
├── 01-cn7-injection-molding/      (구 루트 산재 파일)
│   └── README.md                  신규
├── 02-cnc-machining/              (구 version_2)
│   └── README.md                  경로 갱신 + 구조 표 개편
└── tasks/
```

git 기준 rename 146건 / 수정 23건 / 신규 5건. 이동은 전부 `git mv`라
이력이 보존됐다(`git log --follow` 추적 가능).

### 계획과 달랐던 점

1. **문서 경로 참조 범위가 예상보다 넓었다.** 계획 때는 README 14곳으로
   봤으나 실제로는 스펙·플랜 문서 전반에 `version_2` 참조가 **477곳** 있었다.
   전수 확인 결과 전부 폴더 위치를 가리키는 참조였고("2차 버전"이라는 추상적
   의미로 쓰인 곳 없음), 파일이 실제로 이동했으므로 일괄 치환했다. 날짜·결정·
   결과 수치는 건드리지 않았다.

2. **`.venv` 이동으로 실행 스크립트 shebang이 깨졌다.** `.venv/bin/pytest`가
   옛 절대경로를 가리켜 `uv run pytest`가 실패했다. `uv sync`는 `pyvenv.cfg`가
   유효하면 재생성을 건너뛰므로 고쳐지지 않는다 — `.venv`를 지우고 다시 만들어야
   했다. **venv는 옮기지 말고 지웠다가 재생성하는 게 맞다.**

3. **MLflow DB에 절대경로가 박혀 있었다.** `experiments.artifact_location`,
   `runs.artifact_uri` 등이 `.../version_2/...`를 가리켜 rename으로 깨질 수
   있었다. 다행히 `tracking.py`의 `_repair_stale_artifact_paths()`가 정확히 이
   상황(다른 머신에서 DB 복사)을 처리하도록 이미 구현돼 있어, `configure_tracking()`
   호출만으로 4개 테이블이 자동 복구됐다.

### 검증 결과

| 항목 | 결과 |
|---|---|
| `git status` rename 인식 | 146건 전부 `R` |
| 01번 트랙 `uv run pytest` | **31 passed** |
| 02번 트랙 `uv run pytest` | **100 passed** |
| `version_2` / `docs/superpowers` 잔여 참조 | **0건** |
| MLflow 아티팩트 경로 복구 | 4개 테이블 전부 새 경로 |
| champion 모델 로드 | 성공 |

### 남은 것

- ~~`02-cnc-machining/mlflow.db` 잔여물~~ → **삭제 완료.** run 0개 / 등록모델
  0개인 빈 껍데기임을 확인 후 지웠다(정상 DB `data/mlflow/mlflow.db`는 run 1개 +
  모델 1개 그대로). 잘못된 위치에서 mlflow를 띄웠을 때 생긴 파일로 보인다.
- 커밋은 사용자가 직접 확인 후 진행.
- `experiments` 테이블에 `.../version_2/mlruns/0`(MLflow 기본 Default 실험)
  경로가 남아 있으나, 이 프로젝트가 쓰지 않는 실험이고 해당 디렉터리도 존재하지
  않아 무해하다.
