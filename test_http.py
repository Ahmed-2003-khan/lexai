import httpx
import asyncio

RETRIEVER_URL = "http://retriever:8001"

TEST_QUERIES = [
    "punishment for murder in Pakistan",
    "requirements for a valid contract",
    "bail in non-bailable offences",
    "burden of proof in court",
    "fundamental rights during arrest",
]


async def test_search(query: str, top_k: int = 5):
    payload = {"query": query, "top_k": top_k}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{RETRIEVER_URL}/search", json=payload)

    if response.status_code != 200:
        print(f"ERROR {response.status_code}: {response.text}")
        return

    data = response.json()
    results = data.get("results", [])

    print("\n" + "=" * 60)
    print(f"QUERY : {query}")
    print(f"HITS  : {data.get('total_results', 0)}")
    print("=" * 60)

    for r in results:
        print(f"\n  Rank {r['rank']} | Score: {r['score']:.4f}")
        print(f"  Title : {r['title']}")
        print(f"  Source: {r['source']}  (chunk {r['chunk_index']})")
        print(f"  Type  : {r['doc_type']} | Jurisdiction: {r['jurisdiction']}")
        print(f"  Preview: {r['content_preview'][:200]}...")
        print("  " + "-" * 56)


async def main():
    # Quick health check first
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{RETRIEVER_URL}/health")
            print(f"Health: {r.json()}")
        except Exception as e:
            print(f"Retriever unreachable: {e}")
            return

    for q in TEST_QUERIES:
        await test_search(q)


if __name__ == "__main__":
    asyncio.run(main())