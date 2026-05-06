from prometheus_client import Counter, Histogram, Gauge

# 1. Total queries counter
QUERY_COUNTER = Counter(
    "lexai_total_queries", 
    "Total number of queries processed", 
    ["jurisdiction", "status"]
)

# 2. Query latency histogram
QUERY_LATENCY = Histogram(
    "lexai_query_latency_seconds", 
    "Time taken to process a query end-to-end",
    buckets=[1.0, 3.0, 5.0, 10.0, 30.0]
)

# 3. Critic score histogram
CRITIC_SCORE = Histogram(
    "lexai_critic_score", 
    "Distribution of scores given by the Critic node",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)

# 4. Token usage counter
TOKEN_USAGE = Counter(
    "lexai_token_usage_total", 
    "Total tokens consumed by LLMs", 
    ["model_name"]
)

# 5. Active queries gauge
ACTIVE_QUERIES = Gauge(
    "lexai_active_queries", 
    "Number of queries currently being processed"
)

# 6. Retriever search latency histogram
RETRIEVER_LATENCY = Histogram(
    "lexai_retriever_latency_seconds", 
    "Time taken by DPR engine to fetch documents"
)