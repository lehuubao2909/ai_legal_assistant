import os
import sys
import asyncio

# Add backend to path so we can import things
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from agent import create_legal_agent

async def run_turn(agent, query):
    print(f"\n--- Sending query: '{query}' ---")
    response = await agent.chat(query)
    print("Chat response received!")
    
    raw_text = ""
    try:
        raw_text = response.text()
        if asyncio.iscoroutine(raw_text):
            raw_text = await raw_text
        print("Raw text response:", raw_text)
    except Exception as err:
        print("Error getting raw text:", err)
        
    try:
        structured_data = await response.structured_output()
        if not structured_data and raw_text:
            print("structured_output() is None, attempting regex fallback extraction...")
            import re
            match = re.search(r"(\{[\s\S]*\})", raw_text)
            if match:
                try:
                    from agent import LegalAgentOutput
                    parsed_obj = LegalAgentOutput.model_validate_json(match.group(1))
                    structured_data = parsed_obj.model_dump()
                    print("Regex fallback successfully parsed structured data!")
                except Exception as val_err:
                    print("Regex fallback validation failed:", val_err)
            else:
                print("No JSON block found in raw text.")
                
        print("Final structured output:", structured_data)
    except Exception as err:
        print("Error getting structured output:")
        import traceback
        traceback.print_exc()

async def main():
    agent = create_legal_agent()
    async with agent:
        print("Agent initialized successfully!")
        
        # Turn 1
        await run_turn(agent, "Tôi muốn sa thải một nhân viên.")
        
        # Turn 2
        await run_turn(agent, "Hợp đồng thử việc")

if __name__ == "__main__":
    asyncio.run(main())

