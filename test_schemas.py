import asyncio
import os
import json
from dotenv import load_dotenv
load_dotenv()

from agent import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

async def run_query(runner, session_id, query_text):
    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=query_text)]
    )
    
    print(f"\n--- Running query: '{query_text}' ---")
    final_output = None
    
    async for event in runner.run_async(
        user_id="test-user",
        session_id=session_id,
        new_message=message
    ):
        # Snipping event output or text representation
        if event.author == "StudyPlannerCoordinator" and event.content and event.content.role == "model":
            for part in event.content.parts:
                if part.text:
                    try:
                        # Attempt to parse the structured response from the coordinator
                        final_output = json.loads(part.text)
                        print(f"[{event.author}] Parsed structured response successfully.")
                    except json.JSONDecodeError:
                        pass
        elif event.content and event.content.role == "model":
            for part in event.content.parts:
                if part.text:
                    print(f"[{event.author}]: {part.text[:200]}...")
            
    print(f"Final Structured Output:")
    print(json.dumps(final_output, indent=2) if final_output else "None")
    return final_output

async def main():
    # Setup session service
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="study-planner-agent",
        session_service=session_service,
        auto_create_session=True,
    )
    
    # 1. Test PlannerAgent
    planner_res = await run_query(
        runner, 
        "session-planner", 
        "I want to set up a study schedule for Chemistry and Calculus, spending 10 hours a week total, studying on Monday and Wednesday evenings."
    )
    assert planner_res is not None
    assert planner_res.get("response_type") == "schedule_creation"
    assert planner_res.get("planner_data") is not None
    print("PlannerAgent verification passed!")
    
    # 2. Test PriorityAgent
    priority_res = await run_query(
        runner, 
        "session-priority", 
        "I need to prioritize my subjects. I have a hard Physics exam in 3 days, a medium English essay due in 10 days, and a Chemistry quiz next week."
    )
    assert priority_res is not None
    assert priority_res.get("response_type") == "task_prioritization"
    assert priority_res.get("priority_data") is not None
    print("PriorityAgent verification passed!")

    # 3. Test RecoveryAgent
    recovery_res = await run_query(
        runner, 
        "session-recovery", 
        "I missed my Monday and Tuesday study sessions for history due to being sick. I need a recovery schedule to catch up."
    )
    assert recovery_res is not None
    assert recovery_res.get("response_type") == "missed_tasks_recovery"
    assert recovery_res.get("recovery_data") is not None
    print("RecoveryAgent verification passed!")

    print("\nALL AUTOMATED SCHEMA VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(main())
