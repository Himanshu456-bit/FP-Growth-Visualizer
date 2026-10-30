"""
Unit tests for FP-growth, Association Rules, and Dataset parsers.
"""
import pytest
from pathlib import Path
from fp_growth.fptree import FPTree, FPNode
from fp_growth.miner import FPGrowth
from fp_growth.rules import generate_association_rules
from fp_growth.datasets import (
    parse_multiline_text,
    parse_csv_file,
    export_transactions_to_csv,
    BENCHMARK_DATASETS
)


def test_fp_node_and_tree():
    """Verify FPNode and FPTree basic pointer and count operations."""
    tree = FPTree()
    tree.add_header_item("A", 2)
    tree.add_header_item("B", 2)

    tree.insert_transaction(["A", "B"], count=1)
    tree.insert_transaction(["A", "B"], count=1)

    assert "A" in tree.root.children
    assert tree.root.children["A"].count == 2
    assert "B" in tree.root.children["A"].children
    assert tree.root.children["A"].children["B"].count == 2

    # Check path to root
    b_node = tree.root.children["A"].children["B"]
    assert b_node.get_path_to_root() == ["A"]


def test_textbook_fp_growth():
    """
    Test against Han, Pei & Yin (2000) canonical textbook dataset.
    Transactions:
    1: f, a, c, d, g, i, m, p
    2: a, b, c, f, l, m, o
    3: b, f, h, j, o
    4: b, c, k, s, p
    5: a, f, c, e, l, p, m, n
    Min count = 3.
    Frequent 1-items with count >= 3:
    f: 4, c: 4, a: 3, b: 3, m: 3, p: 3
    """
    transactions = [
        ["f", "a", "c", "d", "g", "i", "m", "p"],
        ["a", "b", "c", "f", "l", "m", "o"],
        ["b", "f", "h", "j", "o"],
        ["b", "c", "k", "s", "p"],
        ["a", "f", "c", "e", "l", "p", "m", "n"],
    ]

    miner = FPGrowth(min_count=3)
    itemsets = miner.fit(transactions)

    # Convert to dictionary for easy assertions: frozenset -> count
    freq_map = {itemset.items: itemset.count for itemset in itemsets}

    # Verify frequent 1-itemsets
    assert freq_map[frozenset(["f"])] == 4
    assert freq_map[frozenset(["c"])] == 4
    assert freq_map[frozenset(["a"])] == 3
    assert freq_map[frozenset(["b"])] == 3
    assert freq_map[frozenset(["m"])] == 3
    assert freq_map[frozenset(["p"])] == 3

    # Verify frequent multi-itemsets
    assert freq_map[frozenset(["f", "c"])] == 3
    assert freq_map[frozenset(["f", "a"])] == 3
    assert freq_map[frozenset(["f", "m"])] == 3
    assert freq_map[frozenset(["c", "a"])] == 3
    assert freq_map[frozenset(["c", "m"])] == 3
    assert freq_map[frozenset(["a", "m"])] == 3
    assert freq_map[frozenset(["f", "c", "a"])] == 3
    assert freq_map[frozenset(["f", "c", "m"])] == 3
    assert freq_map[frozenset(["f", "a", "m"])] == 3
    assert freq_map[frozenset(["c", "a", "m"])] == 3
    assert freq_map[frozenset(["f", "c", "a", "m"])] == 3


def test_association_rules():
    """Verify association rules generation and mathematical properties."""
    transactions = [
        ["Milk", "Bread", "Butter"],
        ["Bread", "Butter"],
        ["Milk", "Bread"],
        ["Milk", "Bread", "Butter"],
    ]
    # Total = 4
    # Milk: 3 (0.75), Bread: 4 (1.0), Butter: 3 (0.75)
    # Milk & Bread: 3 (0.75), Butter & Bread: 3 (0.75), Milk & Butter: 2 (0.5), Milk & Bread & Butter: 2 (0.5)

    miner = FPGrowth(min_support=0.5)
    itemsets = miner.fit(transactions)
    rules = generate_association_rules(itemsets, min_confidence=0.6, min_lift=0.9)

    assert len(rules) > 0
    for r in rules:
        # Confidence = support(A U C) / support(A)
        assert pytest.approx(r.confidence, 0.01) == (r.support / r.antecedent_support)
        # Lift = confidence / support(C)
        assert pytest.approx(r.lift, 0.01) == (r.confidence / r.consequent_support)
        # Leverage = support(A U C) - (support(A) * support(C))
        assert pytest.approx(r.leverage, 0.01) == (r.support - (r.antecedent_support * r.consequent_support))


def test_custom_multiline_text_parser():
    """Verify multiline custom text input parsing."""
    raw_text = """
    # Comments should be ignored
    Apple, Banana, Orange
    Banana, Orange
    
    Apple; Mango; Grape
    """
    txs = parse_multiline_text(raw_text)
    assert len(txs) == 3
    assert txs[0] == ["Apple", "Banana", "Orange"]
    assert txs[1] == ["Banana", "Orange"]
    assert txs[2] == ["Apple", "Mango", "Grape"]


def test_csv_parser_basket_and_long_formats(tmp_path):
    """Verify reading CSVs in both basket format and long format."""
    # 1. Basket format
    basket_csv = tmp_path / "baskets.csv"
    basket_csv.write_text("transaction_id,items\n1,\"Milk, Bread\"\n2,\"Bread, Eggs\"\n", encoding="utf-8")

    txs, fmt = parse_csv_file(basket_csv)
    assert len(txs) == 2
    assert txs[0] == ["Milk", "Bread"]
    assert txs[1] == ["Bread", "Eggs"]

    # 2. Long format
    long_csv = tmp_path / "long.csv"
    long_csv.write_text("order_id,product\nORD1,Laptop\nORD1,Mouse\nORD2,Keyboard\n", encoding="utf-8")

    txs_long, fmt_long = parse_csv_file(long_csv)
    assert len(txs_long) == 2
    assert "Laptop" in txs_long[0] and "Mouse" in txs_long[0]
    assert txs_long[1] == ["Keyboard"]


def test_edge_cases():
    """Verify empty and single-element edge cases."""
    # Empty transactions
    miner = FPGrowth()
    assert miner.fit([]) == []

    # Single item
    miner = FPGrowth(min_support=0.5)
    res = miner.fit([["A"], ["A"], ["B"]])
    assert len(res) >= 1
    # Rules on 1-itemsets only should be empty
    rules = generate_association_rules(res, min_confidence=0.5)
    assert rules == []
