from typing import cast

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName

load_dotenv()

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')


class MyModel(BaseModel):
    city: str
    country: str


model = cast(KnownModelName, 'openai:gpt-4o')
print(f'Using model: {model}')
agent: Agent = Agent(model, result_type=MyModel)

if __name__ == '__main__':
    result = agent.run_sync('The windy city in the US of A.')
    print(result.data)
    print(result.cost())
    print(result.all_messages())