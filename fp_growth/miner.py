from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, FrozenSet
from .fptree import FPNode, FPTree


@dataclass
class FrequentItemset:
    items: FrozenSet[str]
    count: int
    support: float

    @property
    def item_list(self) -> List[str]:
        return sorted(list(self.items))

    def __repr__(self) -> str:
        return f"Itemset({', '.join(self.item_list)} | count={self.count}, support={self.support:.3f})"


@dataclass
class ConditionalPatternBase:
    item: str
    paths: List[Tuple[List[str], int]] = field(default_factory=list)


@dataclass
class MiningTrace:
    raw_item_counts: Dict[str, int] = field(default_factory=dict)
    frequent_1_items: Dict[str, int] = field(default_factory=dict)
    pruned_items: Dict[str, int] = field(default_factory=dict)
    sorted_order: List[str] = field(default_factory=list)
    filtered_transactions: List[List[str]] = field(default_factory=list)
    conditional_pattern_bases: Dict[str, List[Tuple[List[str], int]]] = field(default_factory=dict)
    conditional_trees_info: Dict[str, List[str]] = field(default_factory=dict)


class FPGrowth:

    def __init__(self, min_support: float = 0.2, min_count: Optional[int] = None):
        self.min_support = min_support
        self.min_count = min_count
        self.total_transactions = 0
        self.effective_min_count = 0
        self.frequent_itemsets: List[FrequentItemset] = []
        self.fp_tree: Optional[FPTree] = None
        self.trace: MiningTrace = MiningTrace()

    def fit(self, transactions: List[List[str]]) -> List[FrequentItemset]:
        self.total_transactions = len(transactions)
        if self.total_transactions == 0:
            self.frequent_itemsets = []
            return []

        if self.min_count is not None:
            self.effective_min_count = self.min_count
            self.min_support = self.effective_min_count / self.total_transactions
        else:
            self.effective_min_count = max(1, int(self.min_support * self.total_transactions))

        self.trace = MiningTrace()

        item_counts: Dict[str, int] = defaultdict(int)
        for tx in transactions:
            unique_items = set(tx)
            for item in unique_items:
                item_counts[item] += 1

        self.trace.raw_item_counts = dict(item_counts)

        frequent_items = {
            item: count
            for item, count in item_counts.items()
            if count >= self.effective_min_count
        }
        pruned_items = {
            item: count
            for item, count in item_counts.items()
            if count < self.effective_min_count
        }

        self.trace.frequent_1_items = dict(frequent_items)
        self.trace.pruned_items = dict(pruned_items)

        sorted_items = sorted(
            frequent_items.keys(),
            key=lambda item: (-frequent_items[item], item)
        )
        self.trace.sorted_order = sorted_items
        item_rank = {item: idx for idx, item in enumerate(sorted_items)}

        tree = FPTree()
        for item in sorted_items:
            tree.add_header_item(item, frequent_items[item])

        filtered_transactions = []
        for tx in transactions:
            ordered_tx = [item for item in tx if item in item_rank]
            ordered_tx.sort(key=lambda item: item_rank[item])
            if ordered_tx:
                tree.insert_transaction(ordered_tx, count=1)
                filtered_transactions.append(ordered_tx)

        self.fp_tree = tree
        self.trace.filtered_transactions = filtered_transactions

        patterns: Dict[FrozenSet[str], int] = {}
        self._mine_tree(tree, set(), patterns, self.effective_min_count, self.trace)

        results: List[FrequentItemset] = []
        for itemset_keys, count in patterns.items():
            support = count / self.total_transactions
            results.append(FrequentItemset(items=itemset_keys, count=count, support=support))


        results.sort(key=lambda x: (-x.support, -len(x.items), sorted(list(x.items))))
        self.frequent_itemsets = results
        return self.frequent_itemsets

    def _mine_tree(
        self,
        tree: FPTree,
        prefix: Set[str],
        patterns: Dict[FrozenSet[str], int],
        min_count: int,
        trace: MiningTrace
    ) -> None:
        items_ascending = sorted(
            tree.header_table.keys(),
            key=lambda x: (tree.header_table[x][0], x)
        )

        for item in items_ascending:
            item_count = tree.header_table[item][0]
            new_pattern = prefix | {item}
            frozen_pattern = frozenset(new_pattern)
            patterns[frozen_pattern] = item_count

            nodes = tree.get_nodes_for_item(item)
            prefix_paths: List[Tuple[List[str], int]] = []
            for node in nodes:
                path = node.get_path_to_root()
                if path:
                    prefix_paths.append((path, node.count))

            trace.conditional_pattern_bases[item] = prefix_paths

            cond_tree, cond_frequent_items = self._build_conditional_tree(prefix_paths, min_count)

            if cond_tree is not None and cond_tree.header_table:
                trace.conditional_trees_info[item] = cond_tree.format_tree()
                self._mine_tree(cond_tree, new_pattern, patterns, min_count, trace)

    def _build_conditional_tree(
        self,
        prefix_paths: List[Tuple[List[str], int]],
        min_count: int
    ) -> Tuple[Optional[FPTree], Dict[str, int]]:
        cond_item_counts: Dict[str, int] = defaultdict(int)
        for path, count in prefix_paths:
            for item in path:
                cond_item_counts[item] += count

        cond_frequent = {
            item: count
            for item, count in cond_item_counts.items()
            if count >= min_count
        }

        if not cond_frequent:
            return None, {}

        cond_rank = sorted(cond_frequent.keys(), key=lambda x: (-cond_frequent[x], x))
        rank_map = {item: idx for idx, item in enumerate(cond_rank)}

        cond_tree = FPTree()
        for item in cond_rank:
            cond_tree.add_header_item(item, cond_frequent[item])

        for path, count in prefix_paths:

            filtered_path = [item for item in path if item in cond_frequent]
            filtered_path.sort(key=lambda x: rank_map[x])
            if filtered_path:
                cond_tree.insert_transaction(filtered_path, count=count)

        return cond_tree, cond_frequent
