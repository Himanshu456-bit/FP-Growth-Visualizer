"""
Association Rule Mining from Frequent Itemsets.
Computes Confidence, Lift, Leverage, and Conviction.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Set, FrozenSet, Optional
import math
from .miner import FrequentItemset


@dataclass
class AssociationRule:
    """Represents a discovered association rule: Antecedent -> Consequent."""
    antecedent: FrozenSet[str]
    consequent: FrozenSet[str]
    antecedent_support: float
    consequent_support: float
    support: float          # Support of Antecedent U Consequent
    confidence: float       # P(Consequent | Antecedent) = support(A U C) / support(A)
    lift: float             # confidence / support(C)
    leverage: float         # support(A U C) - (support(A) * support(C))
    conviction: float       # (1 - support(C)) / (1 - confidence)

    @property
    def antecedent_str(self) -> str:
        return ", ".join(sorted(list(self.antecedent)))

    @property
    def consequent_str(self) -> str:
        return ", ".join(sorted(list(self.consequent)))

    @property
    def rule_repr(self) -> str:
        return f"{{{self.antecedent_str}}} => {{{self.consequent_str}}}"

    def __repr__(self) -> str:
        return (
            f"Rule({self.rule_repr} | "
            f"supp={self.support:.3f}, conf={self.confidence:.3f}, lift={self.lift:.3f})"
        )


def generate_association_rules(
    frequent_itemsets: List[FrequentItemset],
    min_confidence: float = 0.5,
    min_lift: float = 1.0
) -> List[AssociationRule]:
    """
    Generate all valid association rules from a list of frequent itemsets.

    Args:
        frequent_itemsets: List of mined FrequentItemset objects.
        min_confidence: Minimum confidence threshold (0.0 to 1.0).
        min_lift: Minimum lift threshold (e.g. 1.0 for positive correlation).

    Returns:
        List of AssociationRule objects meeting the thresholds, sorted by lift descending.
    """
    # Build lookup map: frozenset -> support
    support_lookup: Dict[FrozenSet[str], float] = {
        itemset.items: itemset.support for itemset in frequent_itemsets
    }

    rules: List[AssociationRule] = []

    # Rules require itemsets with length >= 2
    multi_itemsets = [itemset for itemset in frequent_itemsets if len(itemset.items) >= 2]

    for itemset in multi_itemsets:
        items = list(itemset.items)
        itemset_supp = itemset.support

        # Generate non-empty proper subsets as antecedents
        for r in range(1, len(items)):
            for ante_combo in combinations(items, r):
                ante = frozenset(ante_combo)
                cons = itemset.items - ante

                ante_supp = support_lookup.get(ante, 0.0)
                cons_supp = support_lookup.get(cons, 0.0)

                if ante_supp <= 0.0 or cons_supp <= 0.0:
                    continue

                conf = itemset_supp / ante_supp
                if conf < min_confidence:
                    continue

                lift = conf / cons_supp
                if lift < min_lift:
                    continue

                leverage = itemset_supp - (ante_supp * cons_supp)

                if conf >= 1.0:
                    conviction = float("inf")
                else:
                    conviction = (1.0 - cons_supp) / (1.0 - conf)

                rules.append(
                    AssociationRule(
                        antecedent=ante,
                        consequent=cons,
                        antecedent_support=ante_supp,
                        consequent_support=cons_supp,
                        support=itemset_supp,
                        confidence=conf,
                        lift=lift,
                        leverage=leverage,
                        conviction=conviction,
                    )
                )

    # Sort rules primarily by lift descending, then confidence descending, then support descending
    rules.sort(key=lambda r: (-r.lift, -r.confidence, -r.support))
    return rules
