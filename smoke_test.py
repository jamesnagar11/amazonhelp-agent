import sys
sys.path.insert(0, 'd:/langchain/project')
import os
os.chdir('d:/langchain/project')

print('=== Testing all imports ===')
from src.rag.state import GraphState
print('OK: state.py')
from src.rag.memory import list_threads, create_thread, get_thread, trim_messages, build_context_string
print('OK: memory.py')
from src.rag.prompts import GET_INTENT_PROMPT, REWRITE_QUERY_PROMPT, EVAL_RETRIEVER_PROMPT, BRAIN_NODE_PROMPT
print('OK: prompts.py')
from src.rag.nodes import get_intent, brain_node, eval_retriever, route_after_eval_retriever
print('OK: nodes.py')
from src.rag.graph import build_graph, get_compiled_graph
print('OK: graph.py')
from src.utils.vector_store import collection_exists, get_qdrant_client
print('OK: vector_store.py')
from src.utils.llm import get_chat_llm, get_judge_llm, get_embeddings
print('OK: llm.py')

print()
print('=== Testing graph build ===')
g = build_graph()
nodes = list(g.nodes.keys())
print('OK: graph has', len(nodes), 'nodes')
print('Nodes:', nodes)

print()
print('=== Testing memory ===')
import uuid
tid = str(uuid.uuid4())
create_thread(tid, 'Smoke Test Chat')
t = get_thread(tid)
title = t['title']
print('OK: thread created, title =', title)

msgs = [{'role': 'human', 'content': 'hello'}, {'role': 'ai', 'content': 'Hi there!'}]
ctx = build_context_string([], msgs)
print('OK: context built, length =', len(ctx))

print()
print('=== Testing JSON parser ===')
from src.rag.nodes import _parse_json
result = _parse_json('{"intent": "Lost/Missing Package", "intent_score": 7}')
print('OK: parsed JSON:', result)

result2 = _parse_json('```json\n{"response": "test"}\n```')
print('OK: parsed fenced JSON:', result2)

print()
print('=== All tests passed! ===')
