# Workly AI Agent

Workly 백엔드가 호출하는 Python/FastAPI 서비스입니다. 프로젝트 계획과 채팅 변경 요청을 검토 가능한 Task 및 담당자 제안으로 변환합니다.

## 워크플로우

```text
프로젝트 계획 또는 변경 요청
        ↓
Planner: Task 구조화
        ↓
Assignment: 구성원 스킬과 업무량 기반 담당자 추천
        ↓
Validator: Task, 담당자, 스킬, 의존성 검증
        ↓
Monitor: 미배정 항목 정리
        ↓
백엔드에 제안 반환 → 프로젝트 Leader 승인
```

Agent는 제안만 반환합니다. 실제 Task 생성과 수정은 백엔드가 승인 요청을 처리할 때 수행합니다.

## 기술 구성

- Python 3.13
- FastAPI, Uvicorn
- LangChain, LangGraph, OpenAI Chat Models
- 선택 연동: TypeSafe/Jev 담당자 추천

## 로컬 실행

Python 3.13과 OpenAI API 키가 필요합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="..."
uvicorn api:app --host 0.0.0.0 --port 8000
```

선택적으로 `OPENAI_MODEL`, `OPENAI_MAX_RETRIES`(기본 6회), `OPENAI_REQUEST_TIMEOUT_SECONDS`(기본 120초), `TYPESAFE_API_KEY`를 설정할 수 있습니다. 일시적인 OpenAI 요청 제한(429)은 SDK가 제공하는 대기 시간을 사용해 재시도합니다. API 키는 저장소에 커밋하지 마세요.

## Docker 실행

저장소 디렉터리에 `.env` 파일을 만들고 적어도 `OPENAI_API_KEY`를 설정한 뒤 실행합니다.

```bash
docker compose up --build
```

## API

| Method | 경로 | 설명 |
| --- | --- | --- |
| `GET` | `/health` | 서비스 상태 확인 |
| `POST` | `/api/agent/workflow` | 프로젝트 계획 또는 Task 제안 생성 |
| `POST` | `/api/agent/messenger-intent` | 채팅 메시지 의도 분류 |
| `POST` | `/api/agent/extract-skills` | 프로필의 명시적 스킬 추출 |

현재 Agent API 자체에는 사용자 인증이 없습니다. 배포 시 공개 접근을 제한하고 백엔드 서비스만 연결하도록 네트워크와 서비스 설정을 구성하세요.

## 백엔드 연결

백엔드의 `AGENT_BASE_URL`에는 Agent 서비스 루트 주소를 설정합니다. 예를 들어 `https://workly-agent.example.com`을 지정하면 백엔드는 `/api/agent/workflow`를 호출합니다. `/api`를 `AGENT_BASE_URL`에 추가하지 마세요.

- [Back](https://github.com/Workly-Ai-Agent/Back) — 인증, 권한, 데이터베이스, 승인 처리
- [Front](https://github.com/Workly-Ai-Agent/Front) — 사용자 인터페이스
