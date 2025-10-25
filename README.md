# Matrix RAG: The "Fuck You" to Complexity

A **single Python script** that checks matrix linearity, stores results in Redis (or JSON), and answers questions using Ollama-powered RAG.
No n8n. No Airflow. No bullshit.

---

## Features
- **Matrix Linearity Checks**: Verify homogeneity/additivity with decorators.
- **RAG Pipeline**: Ask natural language questions about your matrices (e.g., "Are there non-linear matrices?").
- **CLI**: Save, check, and query matrices with zero friction.
- **Redis/JSON Storage**: Pick your poison.

---

## Setup

### 1. Install Dependencies
```bash
pip install numpy redis scikit-learn ollama
# Linux
sudo apt install redis-server
sudo systemctl start redis-server
```
---

2. Start Redis (or don’t, and use JSON)
```bash
# macOS
brew install redis
brew services start redis
```
---
3. Start Ollama
```bash
ollama serve
```
---

(Keep it running in a separate terminal.)

Save a Matrix
```bash
python3 matrix_rag.py --save
# Enter matrix: [[1,2],[3,4]]
# Describe the matrix: A 2x2 linear matrix.
```

---

Check Linearity
```bash
python3 matrix_rag.py --check
# Enter matrix: [[1,2],[3,4]]
# Enter vector v1: [1,0]
# Enter vector v2: [0,1]
# Enter scalar: 2
```

---

Ask a RAG Query
```bash
python3 matrix_rag.py --rag "Are there non-linear matrices?"
```

---

No Redis? No Problem.
Edit the script to use JSON storage instead (see save_to_json method).

---

Why This Exists

To prove that one script can rule them all.
To avoid dependency hell.
Because you can.

---

License
WTFPL – Do what the fuck you want

---
