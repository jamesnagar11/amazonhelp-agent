### Full Stack RAG app for Customer Support Agent for AmazonHelp

### Goal: In the root folder create a full stack RAG app for customer support agent for AmazonHelp

- First Step is to install all the required dependencies for the app. in the root folder, create a requirements.txt file and install all the required dependencies for the app.

## Component of the app: 
1. Langgraph RAG (Retreival Augmented Generation)
2. Frontend in streamlit
3. Database -> Qdrant
4. LLM Models that are used:
    - For Embedding Model: BAAI/bge-small-en-v1.5
    - For Chat Model + Retrieval: Qwen/Qwen3.8-27B
    - For CRAG LLM as Judge: deepseek-ai/DeepSeek-V3.2

## First we form ingestion pipeline + Indexing of Database
Goal: ingest the sample_amazon_dataset.csv file to qdrant (indexing)
inside ./src/ingestion folder you'll write code related to ingestion pipeline
1. load the sample_amazon_dataset.csv dataset using a CSVLoader from langchain
2. chunk the loaded data using a RecursiveCharacterTextSplitter from langchain.text_splitter
3. embed the chunk data using a EmbeddingModel "BAAI/bge-small-en-v1.5"
4. store the embedded data in qdrant using a Qdrant from langchain.vectorstores

Also write commands to start the ingestion pipeline.

# Building the RAG pipeline (Most Important Part of this Project)
Goal: build a RAG pipeline for Qwen/Qwen3.8-27B
inside ./src/rag folder you'll write code related to RAG pipeline
- Persist the DB using SqliteSaver and checkpoints and provide them when you compile the graph.
- A user can have multiple chats features should be enabled in the RAG app.
Where each chat should be independent of other chats. (i.e. message history should be maintained for each chat)
Each chat conversation of multiple messages is uniquely identified by the thread_id used in config. 
Chats can be started new at any time.Chats can be deleted at any time. By chats here I mean just like chatgpt has multiple chat feature, where each conversations of a chat room is tracked and stored in DB.
You should implement Short Term Memory and also the chats history should be maintained across the sessions.
You also have feature to resume chats from where you left previously by just loading same set of thread_id from DB, and chat history.
A chat will have a sliding window of 10 messages for context and should also have a MAX_LIMIT=some_tokens good metrics for my free hugging face model, where either 10 messages based full chat history is persisted or MAX_LIMIT based on tokens are persisted. And for the rest of history you'll create a function that create and returns the summary of the previous chats using LLM Model of DeepSeek-V3.2. by providing all the previous context in it.
Whenever the new summary is generated then append it to the list of summary state for that chat.
A chat will have Memory of List of Summary and last 10 messages. And the user's current query message and llm responses, also across the summary and messages please maintain the human messages and ai messages in some kind of dictionary structure to persist it in DB and retrieve it. so that the context of the chat history is maintained across the sessions.
Chat features is completed by ensuring each of these rules.

#### Nodes and edges of the whole RAG + CRAG Pipeline flow
## Nodes
- START
- get_intent
- intent_evaluator
- rewrite_query
- human_node
- retriever
- eval_retriever
- correct
- ambiguous
- incorrect
- decompose_strips
- recompose_strips
- rewrite_retriever
- ambiguous_retriever
- recompose_ambiguous_strips
- END

## Edges
- start -> get_intent
- get_intent -> intent_evaluator
- condiditional edge intent_evalutator to following:
    - rewrite_query
    - human_node
- human_node -> END
- rewrite_query -> retriever
- retriever -> eval_retriever
- eval_retriever -> condiditional edge to following:
    - correct
    - ambiguous
    - incorrect
- correct -> brain_node
- brain_node -> END
- ambiguous -> ambiguous_retriever
- ambiguous_retriever -> recompose_ambiguous_strips
- recompose_ambiguous_strips -> human_node
- incorrect -> human_node

## Intents
Intent	Score	Escalate straight to human?
Account Security/Hacked	, Score: 10, Escalate:	Yes, immediately — active fraud/compromise risk, every minute matters
Unauthorized/Incorrect Charge	, Score: 9	, Escalate: Yes, fast-tracked — bot can pull the charge details first, but a human should own the resolution given financial/legal exposure
Customer Service Complaint (Escalation)	, Score: 8	, Escalate: Yes — by definition the customer already wants a human; routing back to a bot usually escalates frustration further
Third-Party Seller Issue	, Score: 7	, Escalate: No — try a mediation flow first, but escalate quickly if the seller is unresponsive or the dispute involves money
Lost/Missing Package	, Score: 7	, Escalate: No — attempt tracking lookup + refund/reship policy first; escalate if abuse-risk flags trip or it's a repeat
Delivered but Not Received	, Score: 6	, Escalate: No — bot can check delivery proof/photo and neighbor-drop-off first
Wrong Delivery Location	, Score: 6	, Escalate: No — usually resolvable via redelivery/refund policy
Refund Status/Delay	, Score: 6	, Escalate: No — status lookup is automatable; escalate only if past SLA
Account Access/Login Issues	, Score: 6	, Escalate: No — self-serve reset first; escalate if resets keep failing (possible compromise)
Damaged Item on Arrival	, Score: 5	, Escalate: No — photo + auto-replace/refund flow works well
Wrong/Incorrect Item Received	, Score: 5	, Escalate: No — same, policy-driven
Missing Item(s) from Order	, Score: 5	, Escalate: No — partial refund/reship is automatable
Return Request/Process	, Score: 4	, Escalate: No — policy-driven, low ambiguity
Order Cancellation	, Score: 4	, Escalate: No — automatable unless order already shipped
Prime Membership - Billing/Trial	, Score: 4	, Escalate: No — policy-driven refund/cancel
Price Discrepancy/Pricing Complaint	, Score: 4	, Escalate: No — price-match policy usually resolves it
Digital Content Access (Prime Video/Music/Kindle)	, Score: 4	, Escalate: No — standard troubleshooting first
Device Technical Support (Echo/Alexa/Fire TV/Kindle)	, Score: 4	, Escalate: No — same
Website/App Technical Issue	, Score: 4	, Escalate: No — same, though flag to engineering if it's a systemic bug
Prime Membership - Cancellation/Benefits	, Score: 3	, Escalate: No — informational/self-serve
Packaging Complaint/Feedback	, Score: 2	, Escalate: No — log as feedback, no resolution needed
Invoice/Documentation Request	, Score: 2	, Escalate: No — pure retrieval task
Product Availability/Stock Inquiry	, Score: 1	, Escalate: No — purely informational
Positive Feedback/Compliment	, Score: 1	, Escalate: No — acknowledge and close

### Formula to calulate the intent
final_score = base_severity + min(iteration_count × 1, 2) + (2 if context_complete_but_unresolved else 0)
If threshold means final_score becomes more than 11 then escalate straight away to human directly.

Now lets walkthrough each component
start is triggered when user hit us with prompt, make sure chat is correctly maintained

- get_intent inputs a query from the state that is maintained across the graph and based on predefined 25 intents that i mentioned above, llm will classify them and return the intent and score using get_intent prompt (make sure you provide all the above list of intents to the llm so that it can classify them correctly). This node will run every time when user hits us with a query. LLM is forced to produce structured output of Intent it has decided and confidence in that Intent. Use PydenticStructuredSchema to for llm with_structure_output to only produce intent: <intent_name> and intent_score: <score_1_to_10>. output will be dict of intent and intent_score and obiviously other states are maintained as it is like query, and current number of iteration, etc. 

- intent_evaluator -> It takes input intent and intent_score and obiviously have the current_iteration in the state of graph and if intent_score is checked if 10 and intent is Account Security/Hacked or Unauthorized/Incorrect Charge or Customer Service Complaint (Escalation) then directly send to Human. else first will calculate final_score based on the formula of final_score and then again if final_score is >= 11 it will send to human else the output will be sent to rewrite_query with these  intent and intent_score as well which might be also considered in the graph state. If it chose to escalate to human then it should also state the reason for which it thinks a human might be needed to alert us about the issue and user query, promblem and why human is needed here , what is llm lacking here , etc. These details will be passed to human_node as input. Also please check that the input format and properties/attributes to be consisntent for human node to receive in the input as incorrect node will also escalate to human_node.

- rewrite_query -> If takes the input query and enhance this query based input query and the intent we now know about and intent score, and also generate a fake short answer for it and in the query itself maintain that it's a fake answer so that later that enchanced query question and the fake answer both are used to retrieve from rag vectore store. llm used here is "deepseek-ai/DeepSeek-V3.2". Use structured out parser of pydantic to only get user_query_rewritten: <enhanced_query> and fake_answer: <fake_answer>. Update this data in the state of the graph.

- retriever -> this node will have user_query_rewritten and fake_answer as input from the state of graph and will use RAG to retrieve context based on these inputs and will return the Top 5 most relavent chunks of context based on the query and fake answer. And this will be send to eval_retriever node. Retriver will get the vector store retriever that we might have made during ingestion pipeline and would be stored in some util.py or related utilities files. The retriever will now have the Top 5 Documents of context chunks retrieved by from vector store qdrant of langchain. And this will be send to eval_retriever node. output will be these Top 5 Documents of context chunks and this will update the state of the graph with these 5 documents in the documents key of the state. And will pass along. But the most important part is we are using the indexing for qdrant, so before retrieving we will check if the index cached already or not and if not then we will create it and if it is cached then we will use it. Keep in mind this idea of indexing. Now this output of 5 documents is saved and will be given to eval_retriever.

- eval_retriever -> This is the evaluation node that will check if the retrieved documents are relevant to the query or not. LLM will check for each of the fetched documents if it's relevant to the query or not and also tell the score (range from 0.0 to 1.0) for each. LLM will Use "deepseek-ai/DeepSeek-V3.2" llm to check the relavence of for each document. and will return the score (range from 0.0 to 1.0) for each of the document. If any one of the document score is more than 0.8 then it will be send to answer node. else if every document score is less than 0.35 then it will send it to incorrect node. if it is between 0.35 and 0.8 then the part which is close to 0.8 means greater than 0.7 is passed to answer node and other documents are passed to ambiguous node. Make sure the llm output is structured with Pydantic Output Parser

- correct -> This node will decompose the input documents into new lines of to form a single document. This single document's each new line is considered as a strip and passed to llm by mentioning all the facts about it explicitly so that the llm "Qwen/Qwen3.8-27B" will generate a relevance score for each of these strips, and the strips with lower score than 0.45 will be removed completely, and the remaining strips will be concatenated. The output of it becomes a single document which is now passed to brain_node.

- incorrect -> This node will receive the output and from previous evaluator node and also has the query , intent , etc. It will be passed to llm "Qwen/Qwen3.8-27B" and get structured output using pydantic output parser. The output should contain the reason stated by llm why it's been escalated, what was the issue of user it can't solve, why it needs a human, what were the relevance in the vector db for this specific task that leads us to incorrect node. This will be the final output for the user which will be passed to the human_node.

- ambiguous -> This node will receive the output and from previous evaluator node and also has the query , intent , etc. In this node we will re-write the prompt with more attention to details in the prompt of re-write prompt so that this time "Qwen/Qwen3.8-27B" can give more relavent information that is missing and might be availabe in database to strictly increase the score. The prompt is based on the parts where the score was less to enchance the score that performed less. The enchanced prompt will give structured output using pydantic schema. And this prompt will be passed to ambiguous_retriever for more relavent information.

- ambiguous_retriever -> this node will receive the output from ambiguous node and will use RAG to retrieve context based on these inputs and will return the Top 4 most relavent chunks of context based on the query and fake answer. And this will be send to eval_retriever node. Retriver will get the vector store retriever that we might have made during ingestion pipeline and would be stored in some util.py or related utilities files. The retriever will now have the Top 4 Documents of context chunks retrieved by from vector store qdrant of langchain. Now, these more relavent 4 document chunks are passed to recompose_ambiguous_strips node.

- recompose_ambiguous_strips -> This node will take these 4 documents and make a single document out of it made up of individual strips which is formed based on pattern matching to make atomic strips. Now, each of these strips is now passed to "Qwen/Qwen3.8-27B" llm to give a score to each strip for their relavance. And discarding the strips whose score is less than 0.45. And finally combine them and make a single document. And the output is now becomes a single document which is now passed to brain_node. And One important thing here is that if none of the scores of any 10% of strips are above 0.4 then it should be passed to human_node with the relavent information. The output should contain the reason stated by llm why it's been escalated, what was the issue of user it can't solve, why it needs a human, what were the relevance in the vector db for this specific task that leads us to incorrect node. This will be the final output for the user which will be passed to the human_node

- brain_node -> Most important node with most attention to detail in the prompt that we create for this node. This node takes the single document from correct or recompose_ambiguous_strips node or both, and provides intent and every other relavent context we have prerved by now so that it can provide the best output by passing this all context into the input of "Qwen/Qwen3.8-27B" llm model. Please make sure that if it doesn't know about the context it should clearly mention it in the output. Make sure it doesn't hallucinate and also be the helpful customer support agent from amazonhelp, based on previously handled history. Draft a reply grounded in how that brand has historically resolved similar issues. The output should be structured. The output of this is passed to END node.

- END -> it's the end of langraph pipeline where the output is returned. It may either be came from human_node or brain_node. If it came from human_node then it should be passed as it is. If it came from brain_node then first save everything in the graph state, and also increase the iteration count to be 1 more than just current iteration count so that if user again write a query for the assitance then it should track the iterations accoss it all. The output of this is not returned back to whoever called it, which is going to be the frontend.

- START -> it will get the prompt input with relavent context from the frontend call and it pass it down to the graph workflow.

### Frontend (Streamlit)

Make engaging frontend from scratch using streamlit.

Use frontend styling like Gemini and Claude mixing by combining both of the world for better user experience.

In this frontend, in the left sidebar we'll have the converstion chats that the user has had with the agent. It should also have a button to start a new chat with the agent.

Keep design style like gemini and claude where user can input his prompt one after other.

The response is streamed instead of printed all at once. so when the llm is processing the prompt it should show that its processing and when the output is generated it should show it in chunks.

In the top right corner we have a dark mode button to switch between dark and light mode.

After the prompt is processed and output is generated, based on the output from brain node the output should be displayed in the chat. The output can be an answer, or if it's generated from the human_node then it should be displayed as it is. It should also tell about why it's displayed and also tell that it will connect a human agent to reach them out as soon as possible or atleast within 24hrs to resolve this issue.

## Expected output is the end-to-end application
With a Readme.md in the root folder to show the steps for the new user to setup locally and ensuring it has from start to end all commands mentioned. From installing dependencies to ingestion to running everything covered and shouldn't be vague but very clear and easy to follow steps for the new user to understand and setup the application, without so much of extra steps or over steps.

Good Luck and finish this task end to end as mentioned.
Please make sure you don't assume anything and state of building it and please ask for the user input's whereever needed.
I have also setup .env so that you can follow along and should also ask for more .env secrets wherever needed.
And also by when you're creating the output file always create a new file with current date and time stamps so that it's easy to track.
And also one more thing make sure you create a .gitignore and also add .env in the .gitignore and also whatever else you think shouldn't be included in the git.

App will be finished in iterations. This iteration should be completed by finishing the app end to end without hallucinations or assumptions and structuring the prompt templates for each phases of grpah node workflow.

Also, if you ever feel like to add on some other features/functions or any other thing that you have done, then you can suggest to make another iteraion in ./agent folder wher you can write the next iteration flow.
Also, if anywhere in this iteration have provided details that are ambiguous or not complete or completely true or contradicts the real architecture, then you can correct it and make a new file with timestamp in ./agent folder wher you can write the next iteration flow. Just make sure to keep a copy of previous version and mention what all things are changed and also what all things are added and what all things are removed. and also mention why it was changed or what was the issue with previous version or anything else.
Make sure you run the project in a virtual environment like python -m venv venv

Good Luck!