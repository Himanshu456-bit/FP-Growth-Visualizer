"""
Streamlit Web Dashboard for FP-Growth Algorithm Showcase.
Interactive FP-Tree visualization, itemset mining, rule analytics,
CSV upload, and custom transaction entry.
"""
from __future__ import annotations
import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from fp_growth.miner import FPGrowth, FrequentItemset
from fp_growth.rules import generate_association_rules, AssociationRule
from fp_growth.datasets import (
    BENCHMARK_DATASETS,
    parse_csv_file,
    parse_multiline_text,
    clean_transaction,
    export_transactions_to_csv,
)
from fp_growth.visualizer import (
    draw_fp_tree,
    draw_association_rules_network,
    draw_rules_scatter,
)

# Page configuration
st.set_page_config(
    page_title="FP Growth Algorithm Visualizer",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for custom transaction builder
if "custom_builder_txs" not in st.session_state:
    st.session_state.custom_builder_txs = [
        ["Milk", "Bread", "Butter"],
        ["Bread", "Butter", "Eggs"],
        ["Milk", "Diapers", "Beer", "Eggs"],
        ["Milk", "Bread", "Diapers", "Beer"],
        ["Bread", "Butter", "Cola"],
        ["Bread", "Milk", "Butter", "Diapers"],
        ["Beer", "Diapers", "Chips"],
        ["Milk", "Bread", "Eggs", "Butter"],
        ["Bread", "Diapers", "Beer", "Cola"],
        ["Milk", "Butter", "Bread", "Eggs"],
    ]

# Sidebar
st.sidebar.title("🌳 FP-Growth Controls")
st.sidebar.markdown("Configure transaction sources & mining parameters.")

data_source = st.sidebar.radio(
    "Data Source",
    options=["Benchmark Presets", "Upload CSV File", "Custom Transaction Builder"],
    index=0
)

transactions = []
source_label = ""

if data_source == "Benchmark Presets":
    dataset_key = st.sidebar.selectbox(
        "Choose Preset Dataset",
        options=list(BENCHMARK_DATASETS.keys()),
        format_func=lambda k: BENCHMARK_DATASETS[k]["name"]
    )
    dataset_meta = BENCHMARK_DATASETS[dataset_key]
    transactions = [list(tx) for tx in dataset_meta["transactions"]]  # type: ignore
    source_label = f"Preset: {dataset_meta['name']}"
    st.sidebar.info(dataset_meta["description"])

elif data_source == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Transactions CSV",
        type=["csv", "txt"],
        help="Upload CSV in Basket format (items per row), Long format (tid, item), or Binary matrix"
    )
    
    # Handy download sample template
    sample_csv = "transaction_id,items\nT001,\"Milk, Bread, Butter\"\nT002,\"Bread, Butter, Eggs\"\nT003,\"Milk, Diapers, Beer\"\n"
    st.sidebar.download_button(
        label="📄 Download Sample CSV",
        data=sample_csv,
        file_name="sample_transactions.csv",
        mime="text/csv"
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        parsed_txs, detected_fmt = parse_csv_file(file_bytes)
        transactions = parsed_txs
        source_label = f"Uploaded File: {uploaded_file.name} ({detected_fmt})"
        st.sidebar.success(f"Detected format: **{detected_fmt}**")
    else:
        st.sidebar.warning("Please upload a CSV file or switch to another data source.")
        transactions = []
        source_label = "No file uploaded"

elif data_source == "Custom Transaction Builder":
    builder_mode = st.sidebar.segmented_control(
        "Input Mode",
        options=["Bulk Multiline Text", "Interactive List Builder"],
        default="Bulk Multiline Text"
    )

    if builder_mode == "Bulk Multiline Text":
        default_text = (
            "Milk, Bread, Butter\n"
            "Bread, Butter, Eggs\n"
            "Milk, Diapers, Beer, Eggs\n"
            "Milk, Bread, Diapers, Beer\n"
            "Bread, Butter, Cola\n"
            "Bread, Milk, Butter, Diapers\n"
            "Beer, Diapers, Chips\n"
            "Milk, Bread, Eggs, Butter\n"
            "Bread, Diapers, Beer, Cola\n"
            "Milk, Butter, Bread, Eggs"
        )
        custom_input = st.sidebar.text_area(
            "Enter Transactions (one per line, comma-separated)",
            value=default_text,
            height=220
        )
        transactions = parse_multiline_text(custom_input)
        source_label = "Custom Bulk Text Input"

    else:
        st.sidebar.markdown("### Add Single Transaction")
        with st.sidebar.form("add_tx_form", clear_on_submit=True):
            new_tx_str = st.text_input("Items (e.g. Laptop, Mouse, Keyboard)")
            submitted = st.form_submit_button("➕ Add Transaction")
            if submitted and new_tx_str.strip():
                items = [x for x in new_tx_str.split(",") if x.strip()]
                cleaned = clean_transaction(items)
                if cleaned:
                    st.session_state.custom_builder_txs.append(cleaned)

        if st.sidebar.button("🗑️ Reset to Default Custom List"):
            st.session_state.custom_builder_txs = [
                ["Milk", "Bread", "Butter"],
                ["Bread", "Butter", "Eggs"],
                ["Milk", "Diapers", "Beer", "Eggs"],
                ["Milk", "Bread", "Diapers", "Beer"],
                ["Bread", "Butter", "Cola"],
            ]
            st.rerun()

        transactions = [list(tx) for tx in st.session_state.custom_builder_txs]
        source_label = f"Interactive Builder ({len(transactions)} transactions)"

# Mining Parameters
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Algorithm Hyperparameters")

total_count = max(1, len(transactions))
min_support = st.sidebar.slider(
    "Minimum Support Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Minimum fraction of transactions an itemset must appear in to be frequent."
)
required_count = max(1, int(min_support * total_count))
st.sidebar.caption(f"Requires at least **{required_count}** out of **{total_count}** transactions.")

min_confidence = st.sidebar.slider(
    "Minimum Confidence Threshold",
    min_value=0.10,
    max_value=1.0,
    value=0.60,
    step=0.05,
    help="Conditional probability P(Consequent | Antecedent)."
)

min_lift = st.sidebar.slider(
    "Minimum Lift Threshold",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help="Ratio of observed support to expected support if antecedent and consequent were independent. Lift > 1 indicates positive association."
)

# Export transactions
if transactions:
    csv_str = export_transactions_to_csv(transactions)
    st.sidebar.download_button(
        label="📥 Download Transactions as CSV",
        data=csv_str,
        file_name="transactions.csv",
        mime="text/csv"
    )

# Title Section
st.title("🌳 FP Growth Algorithm Visualizer")
st.markdown(
    f"Currently analyzing: **{source_label}** | Total Baskets: **{len(transactions)}**"
)

if not transactions:
    st.warning("No transactions found. Please select or input transactions from the sidebar.")
    st.stop()

# Run FP-Growth Mining
miner = FPGrowth(min_support=min_support)
frequent_itemsets = miner.fit(transactions)
rules = generate_association_rules(
    frequent_itemsets,
    min_confidence=min_confidence,
    min_lift=min_lift
)

# Compute basic dataset statistics
all_items = [item for tx in transactions for item in tx]
unique_items = sorted(list(set(all_items)))
avg_basket_size = sum(len(tx) for tx in transactions) / len(transactions)

# Metric KPI Cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Transactions", len(transactions))
c2.metric("Unique Items", len(unique_items))
c3.metric("Avg Basket Size", f"{avg_basket_size:.1f} items")
c4.metric("Frequent Itemsets", len(frequent_itemsets))
c5.metric("Association Rules", len(rules))

st.markdown("---")

# Main Content Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Dataset & Exploratory Analysis",
    "🌳 2. FP-Tree Architecture",
    "⛏️ 3. Frequent Patterns & Mining",
    "🔗 4. Association Rules & Insights"
])

# ----------------- TAB 1: DATASET EXPLORATION -----------------
with tab1:
    st.subheader("Transaction Database Overview")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Transaction Baskets")
        tx_data = [
            {"Transaction ID": f"T{idx+1:04d}", "Basket Size": len(tx), "Items": ", ".join(tx)}
            for idx, tx in enumerate(transactions)
        ]
        df_tx = pd.DataFrame(tx_data)
        st.dataframe(df_tx, use_container_width=True, height=360)

    with col_right:
        st.markdown("#### Item Frequencies (Support Counts)")
        item_counts = pd.Series(all_items).value_counts().reset_index()
        item_counts.columns = ["Item", "Frequency"]
        item_counts["Support %"] = (item_counts["Frequency"] / len(transactions)) * 100

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
        colors = [
            "#38A169" if f >= required_count else "#CBD5E0"
            for f in item_counts["Frequency"]
        ]
        ax.barh(item_counts["Item"], item_counts["Frequency"], color=colors)
        ax.axvline(
            required_count,
            color="#E53E3E",
            linestyle="--",
            label=f"Min Support Cutoff ({required_count} txs)"
        )
        ax.set_xlabel("Frequency (Count)", fontweight="bold")
        ax.set_ylabel("Item", fontweight="bold")
        ax.set_title("Item Frequency (Green = Meets Min Support)", fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.legend(loc="lower right")
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ----------------- TAB 2: FP-TREE ARCHITECTURE -----------------
with tab2:
    st.subheader("FP-Tree Structural Topology & Diagnostics")
    st.markdown(
        "FP-Growth constructs a compact prefix-tree that preserves transaction co-occurrence "
        "without generating billions of candidate sets."
    )

    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.markdown("#### Step 1: 1-Itemset Filtering")
        trace = miner.trace
        diag_rows = []
        for item, count in sorted(trace.raw_item_counts.items(), key=lambda x: -x[1]):
            is_freq = count >= miner.effective_min_count
            diag_rows.append({
                "Item": item,
                "Count": count,
                "Support": f"{(count / len(transactions)):.1%}",
                "Status": "Frequent" if is_freq else "Pruned"
            })
        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, height=260)

        st.markdown("#### Header Table (Descending Order)")
        st.write(" -> ".join(trace.sorted_order))

        with st.expander("📄 View ASCII Hierarchy Tree"):
            if miner.fp_tree:
                st.code("\n".join(miner.fp_tree.format_tree(ascii_only=True)), language="text")

    with col_t2:
        st.markdown("#### FP-Tree Network Graph")
        if miner.fp_tree:
            fig_tree = draw_fp_tree(miner.fp_tree)
            st.pyplot(fig_tree)
            plt.close(fig_tree)

# ----------------- TAB 3: FREQUENT PATTERNS & MINING -----------------
with tab3:
    st.subheader("Frequent Itemsets Discovered")
    st.markdown(
        f"Found **{len(frequent_itemsets)}** itemsets meeting the minimum support threshold of **{min_support:.1%}**."
    )

    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        st.markdown("#### Conditional Pattern Bases")
        st.caption("Prefix paths mined bottom-up from the FP-tree:")
        cpb_data = []
        for item, paths in miner.trace.conditional_pattern_bases.items():
            if paths:
                p_str = ", ".join([f"({', '.join(p[0])} : {p[1]})" for p in paths])
            else:
                p_str = "None (Root child)"
            cpb_data.append({"Mined Item": item, "Prefix Paths": p_str})
        st.dataframe(pd.DataFrame(cpb_data), use_container_width=True, height=350)

    with col_m2:
        st.markdown("#### Mined Frequent Itemsets Table")
        itemset_data = [
            {
                "Itemset": "{" + ", ".join(itemset.item_list) + "}",
                "Length": len(itemset.items),
                "Count": itemset.count,
                "Support": itemset.support,
                "Support %": f"{itemset.support:.1%}"
            }
            for itemset in frequent_itemsets
        ]
        df_itemsets = pd.DataFrame(itemset_data)

        len_filter = st.multiselect(
            "Filter by Itemset Length",
            options=sorted(df_itemsets["Length"].unique()) if not df_itemsets.empty else [],
            default=sorted(df_itemsets["Length"].unique()) if not df_itemsets.empty else []
        )
        if not df_itemsets.empty and len_filter:
            filtered_df = df_itemsets[df_itemsets["Length"].isin(len_filter)]
            st.dataframe(
                filtered_df[["Itemset", "Length", "Count", "Support %"]],
                use_container_width=True,
                height=300
            )

            # Download Itemsets CSV
            csv_itemsets = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Download Frequent Itemsets CSV",
                data=csv_itemsets,
                file_name="frequent_itemsets.csv",
                mime="text/csv"
            )

# ----------------- TAB 4: ASSOCIATION RULES & INSIGHTS -----------------
with tab4:
    st.subheader("Association Rules Mining & Correlation Graph")

    if not rules:
        st.info(
            "No association rules found with current thresholds. "
            "Try lowering the Minimum Confidence or Minimum Support sliders in the sidebar."
        )
    else:
        # Highlight Top Rule Card
        top_rule = rules[0]
        st.success(
            f"🎯 **Key Recommendation:** When a customer purchases **{{{top_rule.antecedent_str}}}**, "
            f"they have a **{top_rule.confidence:.1%}** likelihood of also buying **{{{top_rule.consequent_str}}}** "
            f"(*Lift: {top_rule.lift:.2f}x* above random purchasing behavior)."
        )

        col_r1, col_r2 = st.columns([1, 1])

        with col_r1:
            st.markdown("#### Top Association Rules Table")
            rules_data = [
                {
                    "Antecedent": "{" + r.antecedent_str + "}",
                    "Consequent": "{" + r.consequent_str + "}",
                    "Support": r.support,
                    "Confidence": r.confidence,
                    "Lift": r.lift,
                    "Leverage": r.leverage,
                    "Conviction": r.conviction if r.conviction != float("inf") else 999.0
                }
                for r in rules
            ]
            df_rules = pd.DataFrame(rules_data)
            st.dataframe(
                df_rules.style.format({
                    "Support": "{:.1%}",
                    "Confidence": "{:.1%}",
                    "Lift": "{:.2f}",
                    "Leverage": "{:+.3f}",
                    "Conviction": "{:.2f}"
                }),
                use_container_width=True,
                height=380
            )

            csv_rules = df_rules.to_csv(index=False)
            st.download_button(
                "📥 Download Association Rules CSV",
                data=csv_rules,
                file_name="association_rules.csv",
                mime="text/csv"
            )

        with col_r2:
            st.markdown("#### Metric Tradeoff: Support vs. Confidence")
            fig_scatter = draw_rules_scatter(rules)
            st.pyplot(fig_scatter)
            plt.close(fig_scatter)

        st.markdown("---")
        st.markdown("#### Directed Association Rule Network")
        fig_net = draw_association_rules_network(rules, top_n=15)
        st.pyplot(fig_net)
        plt.close(fig_net)
