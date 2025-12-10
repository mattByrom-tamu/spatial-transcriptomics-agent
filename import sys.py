import sys
import os
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant.")

result = Runner.run_sync(agent, "Write a haiku about recursion.")
print(result.final_output)


