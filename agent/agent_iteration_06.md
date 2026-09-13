## Iteration 06 : (Goal: Integrate Langsmith to this project)

- First step is to install all the required dependencies for the app and mention them as well in requirements.txt

- Install langsmith by running command: pip install -r requirements.txt

- in the .env file, I have provided LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT and I have already added LANGSMITH_TRACING=true in the .env file.

Now your task is to from langsmith import traceable.
Then use this @traceable decorator over the whole pipeline and it's individual function with name, metadata and 1-2 tags each. 

## Expected Output:
Langsmith feature integration in the application only. Do not add or change any other functionality or feature.