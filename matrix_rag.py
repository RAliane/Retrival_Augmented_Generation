import json
import numpy as np
from functools import wraps
import ollama
import redis
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os

# --- Setup ---
r = redis.Redis(host="localhost", port=6379, db=0)  # Redis for embeddings

# --- Decorators ---
def linearity_decorator(func):
    @wraps(func)
    def wrapper(self, matrix, v1, v2, c):
        lhs_homogeneous = matrix @ (c * v1)
        rhs_homogeneous = c * (matrix @ v1)
        is_homogeneous = np.allclose(lhs_homogeneous, rhs_homogeneous)

        lhs_additive = matrix @ (v1 + v2)
        rhs_additive = matrix @ v1 + matrix @ v2
        is_additive = np.allclose(lhs_additive, rhs_additive)

        if not is_homogeneous:
            raise ValueError("Not homogeneous.")
        if not is_additive:
            raise ValueError("Not additive.")
        return func(self, matrix, v1, v2, c)
    return wrapper

# --- Matrix Class ---
class MatrixRAG:
    @linearity_decorator
    def is_linear(self, matrix, v1, v2, c):
        return {"linear": True}

    def check_matrix(self, matrix, v1=None, v2=None, c=None):
        try:
            if v1 is not None:
                result = self.is_linear(matrix, v1, v2, c)
                return {"status": result, "error": None}
            else:
                return {"status": {"homogeneous": True, "additive": True}, "error": None}
        except ValueError as e:
            return {"status": None, "error": str(e)}

    def generate_embedding(self, text):
        response = ollama.embeddings(model="granite-embedding:278m", prompt=text)
        return response["embedding"]

    def save_to_redis(self, matrix, description):
        embedding = self.generate_embedding(description)
        data = {
            "matrix": matrix.tolist(),
            "description": description,
            "embedding": embedding,
        }
        r.set(f"matrix:{description[:20]}", json.dumps(data))

    def semantic_search(self, query, top_k=3):
        query_embedding = self.generate_embedding(query)
        keys = r.keys("matrix:*")
        results = []
        for key in keys:
            data = json.loads(r.get(key))
            similarity = cosine_similarity(
                [query_embedding],
                [data["embedding"]]
            )[0][0]
            results.append((key, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return [json.loads(r.get(key)) for key, _ in results[:top_k]]

    def rag_query(self, query):
        results = self.semantic_search(query)
        context = "\n".join([f"Matrix: {res['matrix']}, Description: {res['description']}" for res in results])
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        response = ollama.generate(model="granite3.1-moe", prompt=prompt)
        return response["response"]

# --- CLI ---
def main():
    parser = argparse.ArgumentParser(description="Matrix Linearity and RAG Tool")
    parser.add_argument("--check", action="store_true", help="Check matrix linearity")
    parser.add_argument("--rag", type=str, help="Ask a RAG query")
    parser.add_argument("--save", action="store_true", help="Save matrix to Redis")
    args = parser.parse_args()

    rag = MatrixRAG()

    if args.save:
        matrix = np.array(eval(input("Enter matrix (e.g., [[1,2],[3,4]]): ")))
        description = input("Describe the matrix: ")
        rag.save_to_redis(matrix, description)
        print("Saved to Redis!")

    elif args.check:
        matrix = np.array(eval(input("Enter matrix: ")))
        v1 = np.array(eval(input("Enter vector v1: ")))
        v2 = np.array(eval(input("Enter vector v2: ")))
        c = float(input("Enter scalar: "))
        result = rag.check_matrix(matrix, v1, v2, c)
        print(json.dumps(result, indent=2))

    elif args.rag:
        answer = rag.rag_query(args.rag)
        print(f"Answer: {answer}")

    else:
        print("Use --check, --rag, or --save. Run with -h for help.")

if __name__ == "__main__":
    main()
