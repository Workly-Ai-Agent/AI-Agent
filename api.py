"""HTTP API for the Workly multi-agent workflow."""
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from workforce_project_agent import classify_message, extract_skills, run

app = FastAPI(title="Workly Workforce Agent", version="1.0.0")

class Employee(BaseModel):
    name: str
    skills: list[str] = Field(default_factory=list)
    current_workload: int = 0

class WorkflowRequest(BaseModel):
    project_name: str
    plan_text: str
    employees: list[Employee] = Field(default_factory=list)

class MessengerRequest(BaseModel):
    message: str

class SkillExtractionRequest(BaseModel):
    profile_text: str

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "UP"}

@app.post("/api/agent/workflow")
def workflow(request: WorkflowRequest) -> dict[str, Any]:
    try:
        result = run(
            {"name": request.project_name, "plan_text": request.plan_text},
            [employee.model_dump() for employee in request.employees],
        )
        return {
            "status": "PENDING_APPROVAL" if result.get("approval_required") else "READY",
            "project_name": result["project_name"], "tasks": result.get("tasks", []),
            "assignments": result.get("assignments", []), "violations": result.get("violations", []),
            "monitoring": result.get("monitoring", []), "approved": result.get("approved", False),
            "log": result.get("log", []),
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

@app.post("/api/agent/messenger-intent")
def messenger_intent(request: MessengerRequest) -> dict[str, Any]:
    return classify_message(request.message).model_dump()

@app.post("/api/agent/extract-skills")
def extract_profile_skills(request: SkillExtractionRequest) -> dict[str, Any]:
    return extract_skills(request.profile_text).model_dump()
