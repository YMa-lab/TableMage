from pathlib import Path

project_dir = Path(__file__).parent.parent.resolve()
import sys

sys.path.append(str(project_dir))

import tablemage as tm

tm.use_agents()
from tablemage.agents import ChatDA_UserInterface

tm.agents.options.set_llm(llm_type="openai", model_name="gpt-5.2")
tm.agents.options.set_multimodal_llm(llm_type="openai", model_name="gpt-5.2")

ChatDA_UserInterface(
    python_only=False,
    tools_only=True,
    multimodal=False,
    tool_rag=True,
    tool_rag_top_k=10,
    memory_size=10000,
).run()
