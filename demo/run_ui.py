import tablemage as tm

tm.use_agents()
from tablemage.agents import ChatDA_UserInterface

tm.agents.options.set_llm(llm_type="openai", model_name="gpt-5.1")

ChatDA_UserInterface(
    python_only=False,
    tool_rag=True,
).run()
