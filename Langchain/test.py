# ==============================
#  logic_lm_rag.py
# ==============================

import os
from dotenv import load_dotenv

# 1. (Optional) If you intend to call OpenAI or another paid LLM,
#    put your API keys into a `.env` file. E.g.:
#    OPENAI_API_KEY=sk-...
#    Then `load_dotenv()` will make them visible via os.getenv().
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------
# 2. SET UP A SMALL “DOCUMENTS” COLLECTION FOR RAG
# ---------------------------------------------------
# We’ll create 10–12 tiny text “docs,” each describing some family facts
# in plain English. In a realistic RAG you’d load PDFs, web pages, etc.

documents = [
    "John is male and Ann is his wife. They have two children: Mary and Bob.",
    "Bob is male, and married to Lisa. Bob and Lisa have two children: Tom and Mark.",
    "Mary is female and married to Sam (not shown in Prolog). Mary and Sam have one child: Susan.",
    "Tom is male and is the son of Bob and Lisa.",
    "Mark is male and is the son of Bob and Lisa.",
    "Susan is female and is daughter of Mary and Mark’s sibling.",
    # We can add some duplicates or more detail if needed:
    "John and Ann are parents of Mary and Bob.",
    "Bob and Lisa are parents of Tom and Mark.",
    "Mary and Sam are parents of Susan.",
    "Tom is a child of Bob and Lisa; Mark is Tom’s brother."
]

# Each “document” will be a tuple (text, id) for vector‐indexing:
docs_for_rag = [ {"page_content": d, "metadata": {"source": f"doc_{i}"}} 
                 for i, d in enumerate(documents) ]


# ---------------------------------------------------
# 3. EMBEDDINGS + FAISS VECTORSTORE
# ---------------------------------------------------
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# a) Initialize a huggingface embedding model (using sentence-transformers under the hood)
#    This does not require any API key; it runs locally.
hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# b) Build a FAISS index on these documents
vector_store = FAISS.from_documents(
    docs_for_rag,
    embedding=hf_embedding
)

# Now `vector_store` can answer similarity searches.


# ---------------------------------------------------
# 4. WRAP PROLOG VIA pyswip
# ---------------------------------------------------
from pyswip import Prolog

# a) Point to your `kb.pl` file so Prolog can consult it
prolog = Prolog()
prolog.consult("kb.pl")  # make sure kb.pl is in the same folder

def run_prolog_query(query_str):
    """
    Run a Prolog query (e.g. "grandparent(X, susan).") 
    against kb.pl. Return a list of variable‐bindings.
    """
    results = list(prolog.query(query_str))
    # e.g. results might be [{"X": "john"}, {"X": "mark"}], etc.
    return results


# ---------------------------------------------------
# 5. SET UP AN LLM TO “TRANSLATE” NL → PROLOG QUERIES
# ---------------------------------------------------
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# You can swap ChatOpenAI(...) for any other LLM wrapper (HuggingFaceHub, LlamaCpp, etc.).
#
# If you do not have an OPENAI_API_KEY, you could switch to:
#   from langchain_community.llms import HuggingFaceHub
#   llm = HuggingFaceHub(huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
#                       repo_id="google/flan-t5-small", model_kwargs={"temperature":0.1})
#
# Below we’ll assume you have `OPENAI_API_KEY` set in .env. If not, replace this block accordingly.

llm = ChatOpenAI(
    temperature=0.0,  # deterministic
    model="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY
)

# a) Prompt template to convert English question → Prolog query
prolog_translation_prompt = PromptTemplate(
    input_variables=["question", "context_snippets"],
    template=
    """
You are a logic‐specialist. 
Below are some relevant context snippets (extracted via RAG) describing family relationships:

{context_snippets}

Translate the following question into a valid Prolog query (using only the predicates defined in kb.pl: 
  male/1, female/1, parent/2, father/2, mother/2, sibling/2, brother/2, sister/2, grandparent/2, 
  ancestor/2, uncle/2, aunt/2, cousin/2). 

Make sure to:
- Write the Prolog query exactly (ending with a period).
- Do not include any commentary—output only the raw Prolog query.

Question: {question}

Prolog query:
"""
)

prolog_translation_chain = LLMChain(
    llm=llm,
    prompt=prolog_translation_prompt,
)


# ---------------------------------------------------
# 6. A “PIPELINE” FUNCTION TO SERVE USER QUERIES
# ---------------------------------------------------
def answer_question_via_rag_and_prolog(user_question: str, k_retrieval: int = 3):
    """
    1) Retrieve top-k context snippets via FAISS.  
    2) Ask the LLM to translate (NL question + context) → Prolog query.  
    3) Run that Prolog query.  
    4) Return the Prolog result(s) to the user.
    """
    # --- 6a) Retrieve top‐k similar documents
    docs_and_scores = vector_store.similarity_search_with_score(user_question, k=k_retrieval)
    # Extract just the text for context
    context_texts = [doc.page_content for doc, _score in docs_and_scores]
    context_snippets = "\n".join(f"- {t}" for t in context_texts)

    # --- 6b) Translate English question → Prolog query
    prolog_query_text = prolog_translation_chain.run(
        question=user_question,
        context_snippets=context_snippets
    ).strip()

    # Ensure it ends with a period
    if not prolog_query_text.endswith("."):
        prolog_query_text += "."

    print(f"\n[DEBUG] Generated Prolog query: {prolog_query_text}")

    # --- 6c) Execute the Prolog query
    results = run_prolog_query(prolog_query_text)

    # --- 6d) Format the output
    if not results:
        return "Prolog returned no solutions."
    else:
        # Each result is a dict, e.g. {"X": "john"}; collect them
        answers = []
        for binding in results:
            # Join all variable‐bindings in one string, e.g. "X = john, Y = susan"
            answers.append(", ".join(f"{var} = {val}" for var, val in binding.items()))
        return "\n".join(answers)


# ---------------------------------------------------
# 7. A SIMPLE CLI LOOP
# ---------------------------------------------------
if __name__ == "__main__":
    print("\n=== LOGIC‐LM + RAG Demo ===")
    print("Type a family‐logic question (e.g. 'Who is the grandfather of susan?') or 'quit' to exit.\n")

    while True:
        user_q = input("Your question> ").strip()
        if user_q.lower() in ("q", "quit", "exit"):
            print("Goodbye!")
            break

        answer = answer_question_via_rag_and_prolog(user_q)
        print(f"\nAnswer(s):\n{answer}\n")
