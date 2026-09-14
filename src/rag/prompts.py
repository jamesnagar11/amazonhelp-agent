"""
All prompt templates for the AmazonHelp CRAG pipeline.
Each prompt is a LangChain PromptTemplate or ChatPromptTemplate.
"""

from langchain_core.prompts import PromptTemplate

# ── Intent list (25 AmazonHelp intents) ─────────────────────────────────────
INTENT_LIST = """
1.  Account Security/Hacked                     (base_score=10)
2.  Unauthorized/Incorrect Charge               (base_score=9)
3.  Customer Service Complaint (Escalation)     (base_score=8)
4.  Third-Party Seller Issue                    (base_score=7)
5.  Lost/Missing Package                        (base_score=7)
6.  Delivered but Not Received                  (base_score=6)
7.  Wrong Delivery Location                     (base_score=6)
8.  Refund Status/Delay                         (base_score=6)
9.  Account Access/Login Issues                 (base_score=6)
10. Damaged Item on Arrival                     (base_score=5)
11. Wrong/Incorrect Item Received               (base_score=5)
12. Missing Item(s) from Order                  (base_score=5)
13. Return Request/Process                      (base_score=4)
14. Order Cancellation                          (base_score=4)
15. Prime Membership - Billing/Trial            (base_score=4)
16. Price Discrepancy/Pricing Complaint         (base_score=4)
17. Digital Content Access                      (base_score=4)
18. Device Technical Support                    (base_score=4)
19. Website/App Technical Issue                 (base_score=4)
20. Prime Membership - Cancellation/Benefits    (base_score=3)
21. Packaging Complaint/Feedback                (base_score=2)
22. Invoice/Documentation Request               (base_score=2)
23. Product Availability/Stock Inquiry          (base_score=1)
24. Positive Feedback/Compliment                (base_score=1)
25. General/Other                               (base_score=3)
"""

# ── get_intent prompt ────────────────────────────────────────────────────────
GET_INTENT_PROMPT = PromptTemplate.from_template(
    """You are an expert Amazon customer support intent classifier specializing in real-world customer service messages and tweets (@AmazonHelp).

TASK:
Analyze the customer query and do TWO things:
  1. Determine if the query should be REJECTED (see REJECTION RULES below).
  2. If NOT rejected, classify it into exactly ONE of the 25 intents from the INTENT LIST.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REJECTION RULES — evaluate these FIRST:
Reject the query (set should_reject=true) if it meets ANY of the following:
  a) Completely unrelated to Amazon, its products, services, or customer support
     (e.g. weather questions, math homework, political opinions, competitor-only topics)
  b) Personal agenda, spam, or promotional content not involving Amazon
  c) Pure gibberish, incomprehensible text, or random characters with no discernible meaning
  d) Offensive content or harassment that has no Amazon support context

Do NOT reject:
  - Frustrated or angry Amazon customers — even if they use strong language
  - Queries that reference Amazon products, orders, accounts, or services in any way
  - Vague queries that could plausibly relate to an Amazon issue
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERNAL REASONING & DISAMBIGUATION RULES (apply ONLY if NOT rejected):
1. Identify the Primary Operational Goal: Determine the core practical action or information the customer needs (e.g., tracking a package, refund status, account access, product issue, billing).
2. Distinguish Operational Issues from Escalations (CRITICAL):
   - Real-world customers frequently use frustrated, angry, or emotional language (e.g., "terrible service", "useless support", "worst experience").
   - Do NOT classify a query as "Customer Service Complaint (Escalation)" simply because the tone is angry or contains complaints about general service.
   - Classify as "Customer Service Complaint (Escalation)" ONLY if the primary issue is specifically about agent/bot misconduct, previous representative failure, or an explicit request to escalate to management where no specific operational item/order/delivery issue can be identified.
   - If an operational issue is present (e.g., package delayed, missing item, wrong charge, refund not received), ALWAYS select the specific operational intent instead of escalation.
3. Evaluate Against All Candidates: Internally compare the query against each candidate intent to find the single most accurate, specific match.
4. Score Alignment: Set "intent_score" to the exact base_score corresponding to the chosen intent as specified in the INTENT LIST.

INTENT LIST (name → base_score):
{intent_list}

CUSTOMER QUERY:
{query}

OUTPUT RULES:
- If should_reject is true:  set intent to "General/Other", intent_score to 1, and provide a short reject_reason explaining why the query is outside Amazon support scope.
- If should_reject is false: set should_reject=false and reject_reason=null; fill intent and intent_score normally.
- Output ONLY the JSON object below — no explanation, no extra text.

Output format (strict JSON):
{{"intent": "<exact intent name from list>", "intent_score": <integer 1-10>, "should_reject": <true|false>, "reject_reason": "<short reason string or null>"}}"""
)

# ── rewrite_query prompt ─────────────────────────────────────────────────────
REWRITE_QUERY_PROMPT = PromptTemplate.from_template(
    """You are an expert Amazon customer support query enhancer.

ORIGINAL QUERY: {query}
DETECTED INTENT: {intent}
INTENT SCORE: {intent_score}

TASK:
1. Rewrite the query to be more specific and searchable for a RAG vector store containing Amazon customer support conversations.
   - Include context clues from the intent
   - Make it verbose and descriptive
2. Generate a SHORT hypothetical answer (fake answer) that WOULD resolve this type of issue.
   - Mark it clearly as a hypothetical/fake answer for retrieval purposes only.
   - Keep it under 80 words.

Return ONLY a JSON object with two keys: "user_query_rewritten" and "fake_answer".
Do NOT add any explanation or extra text.

Output format (strict JSON):
{{"user_query_rewritten": "<enhanced query>", "fake_answer": "<short hypothetical answer>"}}"""
)

# ── eval_retriever prompt ────────────────────────────────────────────────────
EVAL_RETRIEVER_PROMPT = PromptTemplate.from_template(
    """You are a retrieval relevance evaluator for Amazon customer support.

CUSTOMER QUERY: {query}
REWRITTEN QUERY: {user_query_rewritten}
DETECTED INTENT: {intent}

You are given {num_docs} retrieved documents. Score each one from 0.0 to 1.0 based on:
- Direct relevance to the customer's issue
- Presence of actionable resolution steps
- Match to the detected intent category

DOCUMENTS:
{documents_text}

Return ONLY a JSON array of {num_docs} objects, each with "doc_index" (0-based) and "score" (float 0.0-1.0).
Do NOT include any explanation — output ONLY the JSON array.

Output format:
[{{"doc_index": 0, "score": 0.85}}, {{"doc_index": 1, "score": 0.42}}, ...]"""
)

# ── correct (decompose_strips) prompt ────────────────────────────────────────
STRIP_SCORE_PROMPT = PromptTemplate.from_template(
    """You are a factual relevance scorer for Amazon customer support content.

CUSTOMER QUERY: {query}
INTENT: {intent}

Below is a document split into numbered strips (atomic sentences/facts).
Score each strip from 0.0 to 1.0 based on relevance to the customer's query.
Strips with higher scores are more directly useful for answering the query.

STRIPS:
{strips_text}

Return ONLY a JSON array of objects with "strip_index" (0-based) and "score" (float 0.0-1.0).
Do NOT include any explanation.

Output format:
[{{"strip_index": 0, "score": 0.9}}, {{"strip_index": 1, "score": 0.2}}, ...]"""
)

# ── incorrect node prompt ────────────────────────────────────────────────────
INCORRECT_ESCALATION_PROMPT = PromptTemplate.from_template(
    """You are an Amazon customer support escalation analyst.

CUSTOMER QUERY: {query}
INTENT: {intent}
INTENT SCORE: {intent_score}
ITERATION COUNT: {iteration_count}

The automated RAG system FAILED to find relevant information for this customer.
All retrieved documents had very low relevance scores (< 0.35).

DOCUMENT SCORES: {doc_scores}

Write a professional escalation summary that includes:
1. A clear description of the customer's issue
2. Why the automated system could not resolve it (specific gaps in the knowledge base)
3. What a human agent should focus on
4. Urgency level based on the intent score

Return ONLY a JSON object with one key: "escalation_reason".

Output format:
{{"escalation_reason": "<detailed escalation reason>"}}"""
)

# ── ambiguous node (query refinement) prompt ──────────────────────────────────
AMBIGUOUS_REWRITE_PROMPT = PromptTemplate.from_template(
    """You are an expert at refining Amazon customer support queries to improve retrieval.

CUSTOMER QUERY: {query}
ORIGINAL REWRITTEN QUERY: {user_query_rewritten}
INTENT: {intent}
INTENT SCORE: {intent_score}

PROBLEM: The initial retrieval returned documents with mixed relevance.
LOW-SCORING DOCUMENT EXCERPTS (score < 0.7):
{low_score_excerpts}

TASK: Create a more targeted retrieval query that:
1. Focuses on the SPECIFIC GAPS identified in the low-scoring documents
2. Uses more precise Amazon support terminology
3. Targets the exact resolution steps needed for this intent
4. Includes key phrases that would appear in successful resolution conversations

Return ONLY a JSON object with two keys: "refined_query" and "refined_fake_answer".
Do NOT include any explanation.

Output format:
{{"refined_query": "<highly targeted query>", "refined_fake_answer": "<refined hypothetical answer>"}}"""
)

# ── recompose_ambiguous_strips escalation prompt ──────────────────────────────
AMBIGUOUS_ESCALATION_PROMPT = PromptTemplate.from_template(
    """You are an Amazon customer support escalation analyst.

CUSTOMER QUERY: {query}
INTENT: {intent}
INTENT SCORE: {intent_score}

Despite a two-stage retrieval attempt, the system found insufficient relevant information.
Less than 10% of document strips had relevance scores above 0.4.

STRIP SCORES SUMMARY: {strip_scores_summary}

Write a professional escalation summary explaining:
1. The customer's issue in clear terms
2. What was attempted (two-stage RAG retrieval)
3. Why human expertise is specifically needed here
4. Recommended next steps for the human agent

Return ONLY a JSON object with one key: "escalation_reason".

Output format:
{{"escalation_reason": "<detailed escalation reason>"}}"""
)

# ── brain_node (final answer) prompt ──────────────────────────────────────────
BRAIN_NODE_PROMPT = PromptTemplate.from_template(
    """You are an expert Amazon customer support agent responding on behalf of @AmazonHelp.
You have extensive knowledge of Amazon's policies, procedures, and how to resolve customer issues.

=== CUSTOMER CONTEXT ===
QUERY: {query}
DETECTED LANGUAGE: {detected_language}
INTENT: {intent}
INTENT SCORE: {intent_score}
ITERATION: {iteration_count}

=== CONVERSATION HISTORY ===
{conversation_history}

=== RETRIEVED KNOWLEDGE BASE CONTEXT ===
{refined_context}

=== INSTRUCTIONS ===
1. Draft a warm, professional, empathetic response in @AmazonHelp's voice.
2. CRITICAL: Respond in the SAME LANGUAGE as the customer query ({detected_language}). If query is French, respond in French; if Spanish, in Spanish; if German, in German; if Japanese, in Japanese; etc.
3. Ground your answer STRICTLY in the retrieved context above — do NOT hallucinate policies.
4. If the context does not contain enough information to fully resolve the issue, say so clearly and direct the customer to call/chat with Amazon support.
5. Be specific: reference the customer's exact issue, not generic advice.
6. Keep the response concise (under 200 words).
7. End with a polite follow-up offer in the detected language.
8. If this is a repeat issue (iteration > 1), acknowledge the customer's frustration.

IMPORTANT: If you genuinely cannot find the answer in the context provided, say:
"I don't have enough information to resolve this directly, but here's what I recommend: [specific action]" (translated into {detected_language})

Return ONLY a JSON object with one key: "response".

Output format:
{{"response": "<your helpful, grounded, empathetic response in {detected_language}>"}}"""
)

# ── intent_evaluator escalation message ──────────────────────────────────────
INTENT_ESCALATION_PROMPT = PromptTemplate.from_template(
    """You are an Amazon escalation triage analyst.

CUSTOMER QUERY: {query}
INTENT: {intent}
INTENT SCORE: {intent_score}
FINAL SCORE (after formula): {final_score}
ITERATION COUNT: {iteration_count}

This customer is being escalated to a human agent because:
- Their issue is high-severity (score {final_score} ≥ 11 or critical intent)

Write a professional internal escalation alert that includes:
1. Summary of the customer's issue
2. Why immediate human intervention is required
3. What the LLM/bot has NOT been able to provide and why
4. Specific risk factors (fraud, financial, account compromise, etc.)
5. Recommended urgency level

Return ONLY a JSON object with one key: "escalation_reason".

Output format:
{{"escalation_reason": "<detailed escalation alert>"}}"""
)
