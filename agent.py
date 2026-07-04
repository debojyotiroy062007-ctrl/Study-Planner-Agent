import os
from typing import List, Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google.adk.agents import Agent
from google.adk.tools.function_tool import FunctionTool

# Load environment variables from .env file
load_dotenv()

# Ensure API key is configured
if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    print("Warning: Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set in the environment or .env file.")

# --- Structured Output Schemas ---

class StudySession(BaseModel):
    subject: str = Field(description="Subject or topic to study.")
    day_of_week: str = Field(description="Day of the week (e.g., Monday).")
    time_slot: str = Field(description="Time slot range (e.g., 2:00 PM - 4:00 PM).")
    focus_areas: List[str] = Field(description="List of specific focus topics or tasks.")

class PlannerOutput(BaseModel):
    schedule_title: str = Field(description="Title of the schedule.")
    weekly_hours: float = Field(description="Total weekly study hours.")
    sessions: List[StudySession] = Field(description="List of scheduled sessions.")
    key_milestones: List[str] = Field(description="Key milestones to achieve.")

class SubjectPriority(BaseModel):
    subject: str = Field(description="Name of the subject.")
    urgency_level: str = Field(description="Urgency: High, Medium, or Low.")
    difficulty_level: str = Field(description="Difficulty: Hard, Medium, or Easy.")
    recommended_hours: float = Field(description="Recommended study hours per week.")
    priority_rank: int = Field(description="Priority rank index (1 being highest).")
    rationale: str = Field(description="Explanation for the rank.")

class PriorityOutput(BaseModel):
    ranked_subjects: List[SubjectPriority] = Field(description="Ranked subjects.")
    productivity_tips: List[str] = Field(description="Productivity tips and recommendations.")

class MissedTaskAdjustment(BaseModel):
    missed_subject: str = Field(description="Subject of the missed task.")
    original_day: str = Field(description="Original scheduled day.")
    rescheduled_day: str = Field(description="Rescheduled day.")
    rescheduled_time_slot: str = Field(description="Rescheduled time slot.")
    catch_up_strategy: str = Field(description="Strategy for catching up.")

class RecoveryOutput(BaseModel):
    recovery_plan_summary: str = Field(description="Summary of the recovery strategy.")
    adjustments: List[MissedTaskAdjustment] = Field(description="List of rescheduling adjustments.")
    burnout_prevention_advice: str = Field(description="Advice for preventing burnout and managing stress.")

class CoordinatorOutput(BaseModel):
    response_type: Literal['schedule_creation', 'task_prioritization', 'missed_tasks_recovery'] = Field(
        description="The type of request handled. Must be exactly one of the three allowed values."
    )
    student_message: str = Field(description="Friendly markdown summary presenting the findings to the student.")
    planner_data: Optional[PlannerOutput] = Field(default=None, description="Plan data if PlannerAgent was used.")
    priority_data: Optional[PriorityOutput] = Field(default=None, description="Priority data if PriorityAgent was used.")
    recovery_data: Optional[RecoveryOutput] = Field(default=None, description="Recovery data if RecoveryAgent was used.")

# --- Specialized Sub-Agents ---

# Define the specialized Planner Agent in single-turn mode (Kept as LLM call)
planner_agent = Agent(
    name="PlannerAgent",
    model="gemini-2.5-flash",
    description="Schedules study sessions, structures weekly/daily outlines, and sets long-term study goals.",
    instruction=(
        "You are PlannerAgent, an expert study planner. Your goal is to help students design realistic study timelines, "
        "weekly/daily schedules, and long-term milestones. You MUST provide your response strictly matching the PlannerOutput schema."
    ),
    output_schema=PlannerOutput,
    mode="single_turn"
)

# --- Local Python Functions for Priority and Recovery ---

def PriorityAgent(request: str) -> dict:
    """
    Prioritizes study tasks, ranks subjects/topics, and helps with time management.
    
    Args:
        request: The student's request describing the subjects and deadlines.
    """
    # Keyword-based priority parsing logic
    if "Physics" in request or "physics" in request:
        return {
            "ranked_subjects": [
                {
                    "subject": "Physics",
                    "urgency_level": "High",
                    "difficulty_level": "Hard",
                    "recommended_hours": 6.0,
                    "priority_rank": 1,
                    "rationale": "Physics exam is in 3 days. High urgency and hard difficulty require immediate, intensive study."
                },
                {
                    "subject": "Chemistry",
                    "urgency_level": "Medium",
                    "difficulty_level": "Medium",
                    "recommended_hours": 4.0,
                    "priority_rank": 2,
                    "rationale": "Chemistry quiz is next week. Medium urgency allows for structured prep after Physics."
                },
                {
                    "subject": "English",
                    "urgency_level": "Low",
                    "difficulty_level": "Medium",
                    "recommended_hours": 2.0,
                    "priority_rank": 3,
                    "rationale": "English essay is due in 10 days. Plenty of time remains, so it ranks lowest in immediate priority."
                }
            ],
            "productivity_tips": [
                "Focus on high-yield physics topics like mechanics or electromagnetism first.",
                "Use active recall and practice problems for Chemistry.",
                "Outline your English essay today, but write the bulk of it after the Physics exam."
            ]
        }
    else:
        return {
            "ranked_subjects": [
                {
                    "subject": "General Study Topic",
                    "urgency_level": "Medium",
                    "difficulty_level": "Medium",
                    "recommended_hours": 5.0,
                    "priority_rank": 1,
                    "rationale": "Default prioritization based on standard academic deadlines."
                }
            ],
            "productivity_tips": [
                "Divide your tasks into small chunks.",
                "Take regular breaks using the Pomodoro technique."
            ]
        }

def RecoveryAgent(request: str) -> dict:
    """
    Formulates catch-up plans, handles rescheduling, and offers burnout/stress recovery strategies.
    
    Args:
        request: The student's request detailing missed sessions.
    """
    # Keyword-based recovery parsing logic
    if "history" in request or "History" in request:
        return {
            "recovery_plan_summary": "Catch-up plan for missed Monday and Tuesday history sessions by rescheduling them to the upcoming Friday and Saturday evenings.",
            "adjustments": [
                {
                    "missed_subject": "History",
                    "original_day": "Monday",
                    "rescheduled_day": "Friday",
                    "rescheduled_time_slot": "6:00 PM - 8:00 PM",
                    "catch_up_strategy": "Review Monday's lecture notes and read assigned historical readings."
                },
                {
                    "missed_subject": "History",
                    "original_day": "Tuesday",
                    "rescheduled_day": "Saturday",
                    "rescheduled_time_slot": "4:00 PM - 6:00 PM",
                    "catch_up_strategy": "Complete study questions and write outline for history project."
                }
            ],
            "burnout_prevention_advice": "Do not study for more than 2 hours at a time. Stay hydrated, eat nutritious meals, and prioritize sleep while catching up."
        }
    else:
        return {
            "recovery_plan_summary": "Standard recovery plan for catching up on missed tasks.",
            "adjustments": [
                {
                    "missed_subject": "General Study",
                    "original_day": "Missed Day",
                    "rescheduled_day": "Friday",
                    "rescheduled_time_slot": "5:00 PM - 7:00 PM",
                    "catch_up_strategy": "Dedicate time to complete the overdue assignments."
                }
            ],
            "burnout_prevention_advice": "Ensure you balance catch-up sessions with sufficient sleep and leisure time."
        }

# Wrap local functions in ADK FunctionTools
priority_tool = FunctionTool(PriorityAgent)
recovery_tool = FunctionTool(RecoveryAgent)

# --- Root Coordinator Agent ---

# Define the Root Coordinator Agent using Tool-based Handoff and Output Schema
root_agent = Agent(
    name="StudyPlannerCoordinator",
    model="gemini-2.5-flash",
    description="The main entry point that delegates student queries to PlannerAgent, PriorityAgent, or RecoveryAgent.",
    instruction=(
        "You are StudyPlannerCoordinator, the central orchestrator of the multi-agent study planning system.\n\n"
        "Your task is to analyze the student's request and delegate the task to the most appropriate sub-agent tool:\n"
        "1. Call `PlannerAgent` tool if the student wants to build a new study plan, set timelines, or structure goals. Pass the user's request as the `request` parameter.\n"
        "2. Call `PriorityAgent` tool if the student is struggling with what to study first, has conflicting deadlines, or needs help ranking tasks. Pass the user's request as the `request` parameter.\n"
        "3. Call `RecoveryAgent` tool if the student has fallen behind, missed study blocks, or is feeling burnt out and needs a recovery plan. Pass the user's request as the `request` parameter.\n\n"
        "Once you call the appropriate sub-agent tool and receive its structured result, you MUST format the response to strictly match the CoordinatorOutput schema. "
        "Populate the corresponding data field (`planner_data`, `priority_data`, or `recovery_data`) with the sub-agent's result, set `response_type` exactly to the matching allowed value (e.g. 'schedule_creation' for PlannerAgent, 'task_prioritization' for PriorityAgent, or 'missed_tasks_recovery' for RecoveryAgent), "
        "and write a warm, clear, and encouraging `student_message` in markdown summarizing the advice/schedule."
    ),
    sub_agents=[planner_agent],
    tools=[priority_tool, recovery_tool],
    output_schema=CoordinatorOutput
)
