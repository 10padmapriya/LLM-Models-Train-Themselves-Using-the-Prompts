# LLM Models Train Themselves Using the Prompts

I built a system where an AI teaches itself.

You give it documents. It reads them, comes up with its own questions and answers, judges whether those questions and answers are actually good, throws out the bad ones, and saves the good ones as training data. Then it does it again. And again.

No manual labeling. No hand-crafted datasets. The model does the work.

---

## Why I Built This

Creating training data for language models is expensive and slow. You normally need humans to write thousands of prompt-response pairs by hand. I wanted to see if the model could do that job itself — and it turns out, it can do it pretty well.

The idea is simple: if an LLM is good enough to answer questions, it's good enough to generate training questions too. And if you use a stronger model to judge the quality of what a weaker model generated, you can filter out the garbage automatically.

That's the whole system. Retrieve context → generate pairs → judge quality → keep the good ones → repeat.

---

## What It Actually Does

Here's what happens when you run it:

**Step 1 — You give it documents**

I used three sets of documents:
- Notes on machine learning for business (ROI, metrics, use cases)
- Python and data science code references (pandas, scikit-learn, best practices)
- Problem solving and reasoning guides (case studies, analytical frameworks)

**Step 2 — It reads and retrieves**

The system splits your documents into small chunks and stores them in a vector database (ChromaDB). When it needs context, it searches for the most relevant chunks using semantic similarity.

**Step 3 — It generates training pairs**

GPT-4o-mini reads the retrieved context and writes realistic prompt-response pairs. Things like:

> *"How do I handle class imbalance in scikit-learn?"*
> *"What metrics should I show to business stakeholders?"*
> *"Walk me through how to prevent overfitting."*

These sound like real questions a real person would ask.

**Step 4 — A stronger model judges each pair**

GPT-4o reads every generated pair and scores it from 0 to 1 on four things:
- Is the response actually accurate?
- Does it answer the question that was asked?
- Is it clearly written?
- Would a real person find it useful?

Anything scoring below 0.75 gets thrown out.

**Step 5 — The good ones get saved**

Accepted pairs are saved as JSONL files, formatted exactly the way HuggingFace expects for fine-tuning. Each file represents one iteration of the loop.

**Step 6 — Everything gets tracked**

MLflow logs every metric across every iteration — acceptance rates, quality scores, score distributions. A drift detector (KL divergence) watches whether the quality of generated data is shifting over time and raises an alert if it is.

---

## Results From My Run

After 6 iterations across the three domains:

- **33 training pairs generated**
- **32 accepted** (97% acceptance rate)
- **Average quality score: 0.89 / 1.00**
- **1 correctly rejected** — scored 0.48, the judge caught that it was vague and unhelpful
- **No drift detected** — quality stayed consistent across all iterations
- **One perfect score of 1.00** in iteration 2

The system caught its own bad output. That's the point.

---

## The Tech Stack

| Layer | Tool | Why |
|---|---|---|
| LLM framework | LangChain | Chains, retrievers, prompt management |
| Generator model | GPT-4o-mini | Fast and cheap for generating lots of pairs |
| Judge model | GPT-4o | Stronger model = more critical, reliable scores |
| Vector store | ChromaDB | Persistent local storage for embeddings |
| Embeddings | OpenAI text-embedding-3-small | High quality, low cost |
| Experiment tracking | MLflow | Logs every metric, run, and parameter |
| Drift detection | SciPy KL Divergence | Catches distribution shift in quality scores |
| Fine-tuning ready | HuggingFace PEFT + TRL | LoRA/QLoRA when GPU is available |

---

## How to Run It

**1. Clone the repo and set up your environment**

```bash
git clone https://github.com/10padmapriya/LLM-Models-Train-Themselves-Using-the-Prompts.git
cd LLM-Models-Train-Themselves-Using-the-Prompts

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Add your OpenAI API key**

```bash
cp .env.example .env
# Open .env and add your key: OPENAI_API_KEY=sk-...
```

**3. Load the documents into the vector store**

```bash
python ingest_all.py
```

**4. Run the self-training loop**

```bash
python trainer.py
```

This runs 3 iterations by default. You'll see each sample being generated, scored, and either accepted or rejected in real time.

**5. View results in MLflow**

```bash
mlflow ui --port 5000
```

Open `http://127.0.0.1:5000` in your browser. You'll see all your metrics, scores, and runs.

**6. Run the full MLOps monitoring**

```bash
python mlops.py
```

---

## Project Files

```
├── rag.py              — loads docs, chunks them, stores in ChromaDB, retrieves context
├── generator.py        — asks the LLM to generate training pairs from context
├── evaluator.py        — uses a judge LLM to score and filter each pair
├── trainer.py          — runs the full loop, saves datasets to output/
├── mlops.py            — tracks metrics in MLflow, runs drift detection
├── ingest_all.py       — one-time script to load all docs into ChromaDB
├── docs/
│   ├── domain_qa/      — ML for business documents
│   ├── code/           — Python and data science code references
│   └── reasoning/      — problem solving and case studies
└── output/             — generated JSONL training datasets (one file per iteration)
```

---

## What's Next

The pipeline currently generates and filters training data. The last step — actual fine-tuning — requires a GPU. When that's available:

- Set `dry_run=False` in `trainer.py`
- The system will run LoRA fine-tuning via HuggingFace `SFTTrainer` on every accepted batch
- Trained adapter weights get saved to `models/` and registered in MLflow

On a free Google Colab GPU, you can upload the `output/*.jsonl` files and fine-tune a small model like Llama 3.2 3B in about 20 minutes.

---

## What I Learned

A few things surprised me building this:

**The judge matters more than the generator.** Using GPT-4o-mini to generate and GPT-4o to judge meant the system was self-correcting. The generator cast a wide net, the judge enforced quality. This separation is what makes the whole thing work.

**Deduplication is important.** Without removing near-duplicate prompts, the model would generate slight variations of the same question over and over. A simple key-based dedup based on the first 100 characters of each prompt was enough.

**Drift detection is underrated.** In a system that runs automatically, you need something to tell you when things go wrong. KL divergence on quality scores is a simple but effective early warning system.

**The data quality was genuinely good.** I expected the system to need a lot of filtering. It didn't — 97% of generated pairs passed the quality bar. That suggests GPT-4o-mini is actually quite capable of generating realistic training data when given good context to work from.

---

## Cost to Run

The entire pipeline costs about $0.15 to run 6 iterations with 6 samples each. A $5 credit on OpenAI's platform will run this project dozens of times.

---

*Built as a portfolio project to demonstrate RAG, LangChain, synthetic data generation, LLM evaluation, and MLOps in a single end-to-end system.*
