# Frequent Pattern Growth (FP-Growth) Algorithm Showcase 🌳

An intuitive, beginner-friendly Python implementation and interactive showcase of the **FP-Growth (Frequent Pattern Growth)** algorithm and **Market Basket Analysis**.

Unlike classical Apriori, FP-Growth does not generate massive candidate combinations ($2^k$). Instead, it compresses shopping transactions into a compact **FP-Tree** and discovers frequent itemsets recursively in just two passes.

---

## ⚡ Super Easy Quickstart (Recommended for First-Time Users)

Simply run the **one-click launcher**:

```bash
python run.py
```

This presents a friendly menu:
```
============================================================
  🌳 FP-GROWTH ALGORITHM & SHOPPING BASKET ANALYZER 🌳
  Discover what items people frequently buy together!
============================================================

Welcome! How would you like to explore?
  [1] 🌐 Launch Web Dashboard (Recommended - visual & easiest to use)
  [2] 🛒 Quick Demo (Supermarket Groceries in 5 seconds)
  [3] 📂 Analyze a CSV File (Upload or point to your transactions)
  [4] ✍️  Type in Custom Transactions (Step-by-step terminal prompt)
  [5] 🧪 Run Automated Tests (Verify everything works correctly)
  [6] ❌ Exit
```

---

## 🌟 What Makes This Easy & Beginner Friendly?

1. **🔰 Beginner Mode (Default)**:
   - **No confusing jargon**: Complex data mining metrics are translated into plain English:
     - **Support** ➡️ *Popularity (% of baskets containing this combo)*
     - **Confidence** ➡️ *Predictability ("80% of Coffee buyers also buy Croissants")*
     - **Lift** ➡️ *Strength of relationship ("2.1x more likely than random chance")*
   - **Actionable Business Takeaways**: Automatic highlights explaining *why* a rule matters and how to bundle items.
   - **60-Second In-App Guide**: Built-in visual explainer for first-time visitors.

2. **🧺 Easy Ways to Add Transactions**:
   - **Upload a CSV File**:
     - Automatically recognizes Basket format (`"Milk, Bread"`), Long format (`transaction_id, item`), or 0/1 Matrix.
     - Includes a **"Download Sample CSV Template"** button directly inside the app so you never have to guess the format.
   - **Interactive "Click & Build" Basket Creator**:
     - Click buttons like `[+ Milk] [+ Bread] [+ Butter]` or type your own item names, then hit `✅ Save Basket`.
   - **Bulk Multiline Paste**:
     - Paste raw lines of items directly into a text area.
   - **4 Ready-to-use Sample Stores**:
     - 🛒 *Supermarket Groceries* (Milk, Bread, Diapers, Beer)
     - 💻 *Tech & Gaming Store* (Gaming PC, Keyboard, Mouse, Monitor)
     - 🎬 *Movie Streaming* (Sci-Fi, Cyberpunk, Thriller, Action)
     - ☕ *Bakery & Cafe* (Espresso, Croissant, Latte, Avocado Toast)

3. **🔬 Advanced Mode Available on Demand**:
   - Switch anytime with one click in the sidebar to see the **FP-Tree structural network graph**, **ASCII tree hierarchy**, **Conditional Pattern Bases**, and **directed rule graphs**.

---

## 📂 Project Structure

```
fp_growth_showcase/
├── run.py                    # 🌟 Interactive beginner launcher
├── app.py                    # Streamlit Web App (Beginner & Advanced views)
├── cli.py                    # Rich Terminal CLI with formatted tables
├── fp_growth/
│   ├── __init__.py           # Package exports
│   ├── fptree.py             # FPNode & FPTree data structures & tree formatting
│   ├── miner.py              # FPGrowth miner & diagnostic execution trace
│   ├── rules.py              # Association rule engine (Confidence, Lift, Leverage, Conviction)
│   ├── visualizer.py         # NetworkX & Matplotlib FP-Tree and rule graphs
│   └── datasets.py           # Benchmark datasets, CSV parser, text parser, CSV exporter
├── data/
│   ├── supermarket_baskets.csv     # Sample basket format CSV
│   └── electronics_transactions.csv# Sample long format CSV
├── tests/
│   └── test_fp_growth.py     # Automated test suite (6/6 tests passing)
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🖥️ Alternative Ways to Run

### 1. Direct Web Dashboard Launch
```bash
streamlit run app.py
```

### 2. Direct CLI Launch
```bash
# Analyze a sample store:
python cli.py --dataset grocery --min-support 0.3 --min-confidence 0.6

# Analyze a CSV file:
python cli.py --csv data/supermarket_baskets.csv --min-support 0.3

# Type transactions interactively into terminal:
python cli.py --interactive
```

### 3. Run Automated Unit Tests
```bash
python -m pytest tests
```
