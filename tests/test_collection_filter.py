"""Quick test for collection filtering"""
import sys
sys.path.insert(0, ".")

from src.agents.query_analysis import query_analysis_agent
from src.agents.retrieval import retrieval_agent

query = "ماهو اطول حديث في صحيح البخاري؟"
print(f"Query: {query}")

state = {"original_query": query}
analysis = query_analysis_agent(state)
print(f"target_collections: {analysis.get('target_collections')}")

state.update(analysis)
result = retrieval_agent(state)
docs = result.get("retrieved_docs", [])

print(f"\nResults: {len(docs)}")
for d in docs:
    print(f"  Hadith #{d.hadith_id} - {d.collection}")
