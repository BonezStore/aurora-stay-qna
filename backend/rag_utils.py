
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    rules = """
    1. Answer ONLY with facts explicitly stated in the Retrieved Policy Information. Do not use outside knowledge or assumptions.
    2. If the question names a property (for example ASH-MIA Brickell), use only the rule or value listed for that property.
    3. Include every condition, exception, limit, deadline, fee amount, and approval requirement that applies to the question.
    4. When the answer depends on the rate plan, booking channel, or confirmation terms, say so explicitly.
    5. Answer the question directly in the first sentence, then add the supporting conditions. Keep it concise (2 to 6 sentences or short bullets).
    6. Do not include retrieved information that is unrelated to the question.
    7. Never promise a refund, waiver, or exception that the policy says requires manager, Finance, or other approval; state who must approve it.
    8. If the retrieved information is insufficient, reply exactly: "The available policy documents do not contain enough information to answer this question. Please check with the property team or escalate."
    9. If retrieved passages conflict, state the conflict and recommend confirming with the property team.
    10. End with a one-line "Source:" naming the policy document(s) used.
    """

    prompt = f"""
    You are an AI assistant answering questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    {rules}

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
