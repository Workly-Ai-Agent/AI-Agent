"""Workly AI Workforce & Project Agent.

Planner -> Assignment -> Validator -> Re-planner is the planning workflow.
Monitor and Messenger Intent are deliberately code-first so that an LLM never
changes a project without validation and human approval.
"""

import json
import os
from typing import Any, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typesafe_sdk import Choice, TypeSafeClient

load_dotenv()

# 재계획은 한 번만 수행한다. 스킬/가용 인력이 부족한 경우 같은 LLM 호출을
# 반복하지 않고 violations를 반환해 Leader가 확인할 수 있도록 한다.
MAX_REPLAN_ROUNDS = 1


class TaskItem(BaseModel):
    task: str
    description: str = ""
    required_skills: List[str] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    start_at: Optional[str] = None
    due_at: Optional[str] = None
    priority: Literal["LOW", "MEDIUM", "HIGH", "URGENT"] = "MEDIUM"


class TaskPlan(BaseModel):
    project_name: str
    tasks: List[TaskItem]


class AssignmentItem(BaseModel):
    task: str
    assignee: Optional[str] = None
    reason: str


class AssignmentPlan(BaseModel):
    assignments: List[AssignmentItem]


class ExtractedSkill(BaseModel):
    name: str
    evidence: str = ""


class SkillExtraction(BaseModel):
    skills: List[ExtractedSkill] = Field(default_factory=list)


class MessengerIntent(BaseModel):
    intent: Literal["TASK_CHANGE", "STATUS_QUERY", "NEW_TASK", "GENERAL"]
    task_reference: Optional[str] = None
    requested_change: Optional[str] = None
    requires_replanning: bool = False
    confidence: float = Field(ge=0, le=1)


class AgentState(TypedDict, total=False):
    project_name: str
    plan_text: str
    employees: list[dict[str, Any]]
    tasks: list[dict[str, Any]]
    assignments: list[dict[str, Any]]
    violations: list[str]
    monitoring: list[str]
    round: int
    approved: bool
    approval_required: bool
    log: list[str]


def llm():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


def planner_node(state: AgentState) -> AgentState:
    skills = sorted({skill for employee in state["employees"] for skill in employee.get("skills", [])})
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a project task planner. Break the project into 3-12 actionable tasks.
Create concrete tasks with a clear, practical description explaining the goal and expected work, required skills, dependencies, and priority. Set start_at and due_at only when dates or a project schedule are provided; use ISO 8601 local date-time strings. Never invent dates. Return structured output."""),
        ("user", "Project: {name}\nPlan: {plan}\nAvailable skills: {skills}"),
    ])
    result = (prompt | llm().with_structured_output(TaskPlan)).invoke({
        "name": state["project_name"], "plan": state["plan_text"],
        "skills": ", ".join(skills),
    })
    state["tasks"] = [task.model_dump() for task in result.tasks]
    state.setdefault("log", []).append(f"Planner: {len(result.tasks)} tasks created")
    return state


def assignment_node(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assignment agent. Assign each task to the best available employee.
Prioritize skill overlap, then prefer employees with lower current_workload when candidates have similar skills. Always choose one of the provided employees when employees exist.
Explain every decision in reason. Return structured output."""),
        ("user", "Tasks:\n{tasks}\nEmployees:\n{employees}\nValidator feedback:\n{feedback}"),
    ])
    result = (prompt | llm().with_structured_output(AssignmentPlan)).invoke({
        "tasks": json.dumps(state["tasks"], ensure_ascii=False),
        "employees": json.dumps(state["employees"], ensure_ascii=False),
        "feedback": " / ".join(state.get("violations", [])),
    })
    assignments = [item.model_dump() for item in result.assignments]
    state["assignments"] = jev_assignments(state, assignments)
    state.setdefault("log", []).append(f"Assignment: {len(result.assignments)} recommendations created")
    return state


def jev_assignments(state: AgentState, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use Jev for the final candidate choice; keep the LLM result as fallback."""
    if not os.getenv("TYPESAFE_API_KEY") or not state.get("employees"):
        return fallback

    criteria = {
        employee["name"]: f"Match required skills and prefer lower current workload ({employee.get('current_workload', 0)} active tasks)"
        for employee in state["employees"]
    }
    assignments_by_task = {item["task"]: item for item in fallback}

    try:
        with TypeSafeClient() as client:
            for task in state["tasks"]:
                response = client.system_one(
                    state={
                        "task": task,
                        "employees": state["employees"],
                    },
                    questions={
                        "assignee": Choice(
                            instructions="Which employee should own this task?",
                            criteria=criteria,
                        )
                    },
                )
                choice = response.answers["assignee"].choice
                item = assignments_by_task.setdefault(task["task"], {"task": task["task"], "reason": "Jev recommendation"})
                item["assignee"] = choice
                item["reason"] = "Jev recommendation based on skills and workload"
    except Exception as exc:
        state.setdefault("log", []).append(f"Jev unavailable; kept fallback assignment: {exc}")

    return list(assignments_by_task.values())


def validator_node(state: AgentState) -> AgentState:
    employees = {employee["name"]: employee for employee in state["employees"]}
    tasks = {task["task"]: task for task in state["tasks"]}
    load: dict[str, int] = {}
    violations: list[str] = []
    task_list = state.get("tasks", [])
    names = [task.get("task") for task in task_list]
    if len(names) != len(set(names)):
        violations.append("중복 Task 이름이 있습니다.")
    for assignment in state.get("assignments", []):
        task = tasks.get(assignment.get("task"))
        name = assignment.get("assignee")
        if not task:
            violations.append(f"존재하지 않는 Task: {assignment.get('task')}")
            continue
        if not name or name not in employees:
            violations.append(f"배정할 구성원이 없습니다: {name or assignment.get('task')}")
            continue
        required = set(task.get("required_skills", []))
        owned = set(employees[name].get("skills", []))
        if required and not required.intersection(owned):
            violations.append(f"Skill 불일치: {task['task']} -> {name}")
    task_names = set(tasks)
    assigned_names = {item.get("task") for item in state.get("assignments", [])}
    for missing in task_names - assigned_names:
        violations.append(f"담당자 추천이 누락된 Task: {missing}")
    graph = {name: set(task.get("depends_on", [])) for name, task in tasks.items()}
    for name, dependencies in graph.items():
        for dependency in dependencies - task_names:
            violations.append(f"존재하지 않는 선행 Task: {name} -> {dependency}")
    visiting: set[str] = set()
    visited: set[str] = set()

    def has_cycle(name: str) -> bool:
        if name in visiting:
            return True
        if name in visited:
            return False
        visiting.add(name)
        if any(has_cycle(dependency) for dependency in graph.get(name, set()) if dependency in graph):
            return True
        visiting.remove(name)
        visited.add(name)
        return False

    if any(has_cycle(name) for name in graph if name not in visited):
        violations.append("Task 의존성 순환이 발견되었습니다.")
    state["violations"] = violations
    state["approved"] = not violations
    state["approval_required"] = True
    state.setdefault("log", []).append(f"Validator: {'approved' if state['approved'] else f'{len(violations)} violations'}")
    return state


def monitor_node(state: AgentState) -> AgentState:
    assigned = {item.get("task") for item in state.get("assignments", [])}
    unassigned = [task["task"] for task in state.get("tasks", []) if task["task"] not in assigned]
    state["monitoring"] = [f"Unassigned task: {task}" for task in unassigned]
    state.setdefault("log", []).append(f"Monitor: {len(unassigned)} unassigned tasks")
    return state


def route(state: AgentState) -> str:
    if state.get("approved") or state.get("round", 0) >= MAX_REPLAN_ROUNDS:
        return "monitor"
    state["round"] = state.get("round", 0) + 1
    return "assignment"


def build_workflow():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("assignment", assignment_node)
    graph.add_node("validator", validator_node)
    graph.add_node("monitor", monitor_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "assignment")
    graph.add_edge("assignment", "validator")
    graph.add_conditional_edges("validator", route, {"assignment": "assignment", "monitor": "monitor"})
    graph.add_edge("monitor", END)
    return graph.compile()


def classify_message(message: str) -> MessengerIntent:
    """Use the optional jev/typesafe classifier, with a deterministic fallback."""
    try:
        from typesafe_sdk import Choice, TypeSafeClient
        response = TypeSafeClient().system_one(
            state=message,
            questions={
                "intent": Choice(
                    instructions="Classify the user's message intent.",
                    criteria={
                        "TASK_CHANGE": "Change, update, reschedule, reassign, or otherwise modify an existing task or plan",
                        "NEW_TASK": "Create or add a new task",
                        "STATUS_QUERY": "Ask for task or project status without requesting changes",
                        "GENERAL": "General discussion or a message unrelated to task management",
                    },
                ),
            },
        )
        intent = response.answers["intent"].choice
        return MessengerIntent(
            intent=intent,
            requested_change=message if intent in ("TASK_CHANGE", "NEW_TASK") else None,
            requires_replanning=intent == "TASK_CHANGE",
            confidence=0.8,
        )
    except Exception:
        lowered = message.lower()
        change_words = ("변경", "수정", "미뤄", "일정", "change", "delay", "move")
        new_words = ("추가", "생성", "새 업무", "new task", "create")
        if any(word in lowered for word in new_words):
            return MessengerIntent(intent="NEW_TASK", requested_change=message, confidence=0.65)
        if any(word in lowered for word in change_words):
            return MessengerIntent(intent="TASK_CHANGE", requested_change=message, requires_replanning=True, confidence=0.65)
        if "상태" in lowered or "status" in lowered:
            return MessengerIntent(intent="STATUS_QUERY", confidence=0.7)
        return MessengerIntent(intent="GENERAL", confidence=0.55)


def extract_skills(text: str) -> SkillExtraction:
    if not text.strip():
        return SkillExtraction()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract explicit professional skills from the user's profile text. Return concise, reusable skill names, deduplicate synonyms, and include a short evidence phrase copied or paraphrased from the input. Do not infer skills without evidence. Return structured output."""),
        ("user", "Profile text:\n{text}"),
    ])
    return (prompt | llm().with_structured_output(SkillExtraction)).invoke({"text": text})


def run(project: dict[str, Any], employees: list[dict[str, Any]]) -> dict[str, Any]:
    initial: AgentState = {
        "project_name": project["name"], "plan_text": project["plan_text"],
        "employees": employees,
        "tasks": [], "assignments": [], "violations": [], "monitoring": [],
        "round": 0, "approved": False, "approval_required": True, "log": [],
    }
    return build_workflow().invoke(initial)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY를 설정한 뒤 실행하세요.")
    with open("projects_sample.json", encoding="utf-8") as file:
        project = json.load(file)[0]
    with open("employees_sample.json", encoding="utf-8") as file:
        employees = json.load(file)
    print(json.dumps(run(project, employees), ensure_ascii=False, indent=2))
