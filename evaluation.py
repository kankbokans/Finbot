# -*- coding: utf-8 -*-
"""
Finbot RAG Evaluation
=====================

Two-layer evaluation of Finbot against eval_queries.json:
1. RETRIEVAL LAYER: Precision, Recall, F1 Score (chapter-level, @3)
2. GENERATION LAYER: Groundedness, Response Completeness (LLM-as-judge)
3. Evaluation report with failure patterns and a release gate
"""

import json
import os
from urllib.request import Request, urlopen

import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

print("="*80)
print("FINBOT RAG EVALUATION: TWO-LAYER APPROACH")
print("="*80)

# ============================================================================
# SETUP: Load Finbot's RAG pipeline
# ============================================================================
# finbot.py can't be imported (it launches Gradio at import time), so this
# mirrors its pipeline: same sitemap, splitter, embeddings, LLM and QA prompt.

with open('eval_queries.json', 'r', encoding='utf-8') as f:
    eval_queries = json.load(f)
print(f"\n✓ Loaded {len(eval_queries)} evaluation queries with ground truth")

# Share finbot.py's saved vector database so the evaluation scores the exact
# index the chatbot serves. Delete chroma_db to rebuild it.
persist_directory = "chroma_db"

if os.path.isdir(persist_directory):
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
    print(f"✓ Loaded vector database from {persist_directory}")
else:
    req = Request(url="https://zerodha.com/varsity/chapter-sitemap2.xml", headers={"User-Agent": "Mozilla/5.0"})
    xml = BeautifulSoup(urlopen(req), "lxml-xml")
    urls = [url.find("loc").text for url in xml.find_all("url")]

    docs = []
    for i, url in enumerate(urls):
        docs.extend(WebBaseLoader(url).load())
        if i % 10 == 0:
            print("Loaded document index:", i)

    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory=persist_directory)
    print(f"✓ Vector store built and saved: {len(docs)} chapters, {len(splits)} chunks")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
system_prompt = (
    "You are a financial assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you don't know."
    "Use three sentences maximum and keep the answer concise. "
    "If the question is not clear ask follow up questions. "
    "\n\n"
    "{context}"
    )
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
# With an empty chat history, Finbot's history-aware retriever passes the
# question straight to the retriever, so a plain retriever behaves the same.
rag_chain = create_retrieval_chain(vectorstore.as_retriever(), create_stuff_documents_chain(llm, qa_prompt))
print("✓ RAG chain ready")

# ============================================================================
# PART 2: RETRIEVAL LAYER EVALUATION
# ============================================================================
print("\n" + "="*80)
print("PART 2: RETRIEVAL LAYER EVALUATION")
print("="*80)
print("\nMetrics: Precision, Recall, F1 Score")
print("Measures: Are we retrieving the right Varsity chapters?")

def retrieve_doc_ids(query, k=3):
    """
    Return the top-k unique chapter URLs for a query.

    Several chunks can come from the same chapter, so chunks are deduplicated
    by their source URL to score chapters rather than chunks.
    """
    chunks = vectorstore.similarity_search(query, k=k * 5)
    doc_ids = []
    for chunk in chunks:
        source = chunk.metadata['source']
        if source not in doc_ids:
            doc_ids.append(source)
    return doc_ids[:k]

def calculate_retrieval_metrics(retrieved_ids, relevant_ids, k=3):
    """
    Calculate Precision, Recall, and F1 at k

    Precision@k = (relevant docs in top-k) / k
    Recall@k = (relevant docs in top-k) / (total relevant docs)
    F1@k = harmonic mean of Precision and Recall
    AP@k = sum of Precision@i at each rank i holding a relevant doc,
           divided by min(k, total relevant docs); averaged over queries = MAP@k
    """
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    # True positives: relevant docs that were retrieved
    tp = len(retrieved_set & relevant_set)

    precision = tp / k if k > 0 else 0
    recall = tp / len(relevant_set) if len(relevant_set) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Average Precision rewards ranking relevant docs higher; a perfect ranking
    # scores 1.0 regardless of how many docs are relevant.
    hits, ap_sum = 0, 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in relevant_set:
            hits += 1
            ap_sum += hits / i
    average_precision = ap_sum / min(k, len(relevant_set)) if len(relevant_set) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'average_precision': average_precision,
        'true_positives': tp,
        'retrieved_count': k,
        'relevant_count': len(relevant_set)
    }

print("\nEvaluating retrieval for all queries...")
retrieval_results = []

for idx, eval_query in enumerate(eval_queries, 1):
    query = eval_query['question']
    relevant_ids = eval_query['relevant_doc_ids']

    print(f"  [{idx}/{len(eval_queries)}] Processing query...", end='\r')

    try:
        retrieved_ids = retrieve_doc_ids(query, k=3)

        metrics = calculate_retrieval_metrics(retrieved_ids, relevant_ids, k=3)
        metrics['query_id'] = eval_query['query_id']
        metrics['question'] = query
        metrics['retrieved'] = retrieved_ids
        metrics['relevant'] = relevant_ids

        retrieval_results.append(metrics)
    except Exception as e:
        print(f"\n  ⚠ Error on query {idx}: {str(e)[:50]}")
        continue

print(f"\n✓ Completed retrieval evaluation for {len(retrieval_results)} queries")

print("\n" + "-"*80)
print("RETRIEVAL METRICS (Averaged across all queries)")
print("-"*80)

avg_precision = np.mean([r['precision'] for r in retrieval_results])
avg_recall = np.mean([r['recall'] for r in retrieval_results])
avg_f1 = np.mean([r['f1'] for r in retrieval_results])
avg_map = np.mean([r['average_precision'] for r in retrieval_results])

print(f"\nMAP@3:       {avg_map:.4f}")
print(f"  → Are relevant chapters found and ranked near the top?")
print(f"  → Target: > 0.80 for production (used by the release gate)")

print(f"\nPrecision@3: {avg_precision:.4f}")
print(f"  → What % of retrieved chapters are actually relevant?")
print(f"  → Capped below 1.0 when a query has fewer than 3 relevant chapters")

print(f"\nRecall@3:    {avg_recall:.4f}")
print(f"  → What % of all relevant chapters did we find?")
print(f"  → Target: > 0.70 for production")

print(f"\nF1 Score@3:  {avg_f1:.4f}")
print(f"  → Balanced measure of retrieval quality")
print(f"  → Capped by Precision@3, so reported for reference only")

print("\n" + "-"*80)
print("PER-QUERY RETRIEVAL RESULTS (Sample)")
print("-"*80)

for result in retrieval_results[:3]:
    print(f"\n{result['query_id']}: {result['question'][:60]}...")
    print(f"  Relevant:  {result['relevant']}")
    print(f"  Retrieved: {result['retrieved']}")
    print(f"  AP: {result['average_precision']:.2f} | Precision: {result['precision']:.2f} | Recall: {result['recall']:.2f} | F1: {result['f1']:.2f}")

# ============================================================================
# PART 3: GENERATION LAYER EVALUATION
# ============================================================================
print("\n" + "="*80)
print("PART 3: GENERATION LAYER EVALUATION")
print("="*80)
print("\nMetrics: Groundedness, Response Completeness")
print("Measures: Is Finbot's answer faithful and complete?")

def parse_score(output):
    """Extract 'Score: X' (0-10) from a judge response and normalize to 0-1."""
    try:
        score_line = [line for line in output.split('\n') if line.startswith('Score:')][0]
        return float(score_line.split(':')[1].strip()) / 10.0
    except (IndexError, ValueError):
        return 0.5  # Default if parsing fails

def evaluate_groundedness(answer, context_docs):
    """
    Groundedness (Faithfulness): Is the answer supported by the retrieved context?
    Uses LLM-as-judge to check if answer contains hallucinations

    Returns: score 0.0-1.0 (higher = more grounded)
    """
    # Judge receives ONLY retrieved evidence to test faithfulness.
    context = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = f"""Evaluate if the ANSWER is fully supported by the CONTEXT. Check for hallucinations or unsupported claims.

CONTEXT:
{context}

ANSWER:
{answer}

Rate the groundedness from 0 to 10:

Provide:
1. Score (0-10)
2. Reasoning (one sentence)

Format:
Score: X
Reasoning: <explanation>"""

    try:
        output = llm.invoke(prompt).content
        score = parse_score(output)
        return {
            'score': score,
            'verdict': 'GROUNDED' if score >= 0.7 else 'PARTIAL' if score >= 0.4 else 'HALLUCINATED',
            'explanation': output
        }
    except Exception as e:
        print(f"\n    ⚠ Error evaluating groundedness: {str(e)[:50]}")
        return {'score': 0.5, 'verdict': 'ERROR', 'explanation': str(e)}

def evaluate_completeness(question, answer, reference_answer):
    """
    Response Completeness: Does the answer fully address the question,
    compared to the reference answer?
    Uses LLM-as-judge

    Returns: score 0.0-1.0 (higher = more complete)
    """
    prompt = f"""Evaluate if the ANSWER fully addresses the QUESTION compared to the REFERENCE ANSWER.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

GENERATED ANSWER:
{answer}

Rate the completeness from 0 to 10:

Provide:
1. Score (0-10)
2. Reasoning (one sentence)

Format:
Score: X
Reasoning: <explanation>"""

    try:
        output = llm.invoke(prompt).content
        score = parse_score(output)
        return {
            'score': score,
            'verdict': 'COMPLETE' if score >= 0.7 else 'PARTIAL' if score >= 0.4 else 'INCOMPLETE',
            'explanation': output
        }
    except Exception as e:
        print(f"\n    ⚠ Error evaluating completeness: {str(e)[:50]}")
        return {'score': 0.5, 'verdict': 'ERROR', 'explanation': str(e)}

print("\nEvaluating generation quality for all queries...")
print("Note: This uses LLM-as-judge (gpt-4o-mini) to evaluate quality\n")

generation_results = []

for idx, eval_query in enumerate(eval_queries, 1):
    query = eval_query['question']

    print("-"*80)
    print(f"\n[{idx}/{len(eval_queries)}] {eval_query['query_id']}: {query}")

    try:
        result = rag_chain.invoke({"input": query, "chat_history": []})
        answer = result['answer']
        print(f"\nFinbot Answer:\n{answer}")

        groundedness = evaluate_groundedness(answer, result['context'])
        print(f"\n  Groundedness: {groundedness['score']:.2f} - {groundedness['verdict']}")

        completeness = evaluate_completeness(query, answer, eval_query['reference_answer'])
        print(f"  Completeness: {completeness['score']:.2f} - {completeness['verdict']}")

        generation_results.append({
            'query_id': eval_query['query_id'],
            'question': query,
            'answer': answer,
            'groundedness_score': groundedness['score'],
            'completeness_score': completeness['score']
        })
    except Exception as e:
        print(f"\n  ✗ Error processing query: {str(e)[:80]}")
        continue

print(f"\n✓ Completed generation evaluation for {len(generation_results)} queries")

print("\n" + "="*80)
print("GENERATION METRICS (Averaged)")
print("="*80)

avg_groundedness = np.mean([r['groundedness_score'] for r in generation_results])
avg_completeness = np.mean([r['completeness_score'] for r in generation_results])

print(f"\nGroundedness:  {avg_groundedness:.4f}")
print(f"  → Are answers supported by retrieved context?")
print(f"  → Target: > 0.80 (minimize hallucinations)")

print(f"\nCompleteness:  {avg_completeness:.4f}")
print(f"  → Do answers fully address the questions?")
print(f"  → Target: > 0.75 (comprehensive responses)")

# ============================================================================
# PART 4: Comprehensive Evaluation Report
# ============================================================================
print("\n" + "="*80)
print("PART 4: COMPREHENSIVE EVALUATION REPORT")
print("="*80)

print(f"""
┌─────────────────────────────────────────────────────────────┐
│                  FINBOT RAG EVALUATION                      │
├─────────────────────────────────────────────────────────────┤
│ Dataset: {len(eval_queries)} evaluation queries                           │
├─────────────────────────────────────────────────────────────┤
│ RETRIEVAL LAYER                                             │
│   • MAP@3:        {avg_map:.4f}  {'✓' if avg_map >= 0.80 else '⚠' if avg_map >= 0.70 else '✗'}                              │
│   • Recall@3:     {avg_recall:.4f}  {'✓' if avg_recall >= 0.70 else '⚠' if avg_recall >= 0.60 else '✗'}                              │
│   • Precision@3:  {avg_precision:.4f}  (reference only)                   │
│   • F1 Score@3:   {avg_f1:.4f}  (reference only)                   │
├─────────────────────────────────────────────────────────────┤
│ GENERATION LAYER                                            │
│   • Groundedness: {avg_groundedness:.4f}  {'✓' if avg_groundedness >= 0.80 else '⚠' if avg_groundedness >= 0.70 else '✗'}                              │
│   • Completeness: {avg_completeness:.4f}  {'✓' if avg_completeness >= 0.75 else '⚠' if avg_completeness >= 0.65 else '✗'}                              │
└─────────────────────────────────────────────────────────────┘

INTERPRETATION:
""")

# ── Failure Pattern Playbook ─────────────────────────────────────────────────
if avg_map >= 0.75 and avg_recall < 0.60:
    print("⚠ PATTERN A — High Precision, Low Recall")
    print("  Retrieval is too conservative: correct chapters found but many missed.")
    print("  Fix: Increase k, use smaller chunks, expand query with synonyms.")

if avg_map < 0.60 and avg_recall >= 0.75:
    print("⚠ PATTERN B — Low Precision, High Recall")
    print("  Too many irrelevant chapters entering the result set.")
    print("  Fix: Use MMR, add metadata filters, improve embeddings.")

if avg_map >= 0.75 and (avg_groundedness < 0.70 or avg_completeness < 0.65):
    print("⚠ PATTERN C — Strong Retrieval, Weak Generation")
    print("  Right chapters retrieved but the LLM is not using them well.")
    print("  Fix: Relax the three-sentence limit, add few-shot examples, use a stronger LLM.")

if avg_groundedness < 0.70:
    print("⚠ PATTERN D — Low Groundedness (Hallucination Risk)")
    print("  LLM is generating claims not supported by retrieved context.")
    print("  Fix: Stricter grounding prompt, require chapter citations.")

# ── Release Gate ──────────────────────────────────────────────────────────────
# MAP@3 replaces Precision@3 and F1@3, whose maximums fall below their
# thresholds when most queries have a single relevant chapter.
thresholds = {
    'map':          {'pass': 0.80, 'review': 0.70},
    'recall':       {'pass': 0.70, 'review': 0.60},
    'groundedness': {'pass': 0.85, 'review': 0.75},
    'completeness': {'pass': 0.75, 'review': 0.65},
}
current_metrics = {
    'map': avg_map, 'recall': avg_recall,
    'groundedness': avg_groundedness, 'completeness': avg_completeness,
}
red_flags, yellow_flags = [], []
for metric, value in current_metrics.items():
    t = thresholds[metric]
    if value < t['review']:
        red_flags.append(f"{metric} = {value:.2f} (min {t['review']})")
    elif value < t['pass']:
        yellow_flags.append(f"{metric} = {value:.2f} (target {t['pass']})")

if red_flags:
    decision = "BLOCK  — Do not deploy. Fix critical metrics first."
elif yellow_flags:
    decision = "REVIEW — Investigate before deploying."
else:
    decision = "PASS   — All metrics in target range. Ready to deploy."

print("\n" + "-"*60)
print("RELEASE GATE DECISION")
print("-"*60)
print(f"  {decision}")
if red_flags:
    print("\n  Critical (RED):")
    for f in red_flags:
        print(f"    ✗ {f}")
if yellow_flags:
    print("\n  Warning (YELLOW):")
    for f in yellow_flags:
        print(f"    ⚠ {f}")
print("-"*60)
