"""
Command Line Interface (CLI) for FP-Growth Algorithm Showcase.
Rich terminal visualizations, step-by-step diagnostic breakdown,
CSV import, and custom transaction entry.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Reconfigure stdout/stderr for Windows console unicode safety
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich import box

from fp_growth.miner import FPGrowth
from fp_growth.rules import generate_association_rules
from fp_growth.datasets import (
    BENCHMARK_DATASETS,
    parse_csv_file,
    parse_multiline_text,
    clean_transaction,
    export_transactions_to_csv
)

console = Console()


def display_welcome_banner():
    console.print(
        Panel.fit(
            "[bold cyan]FP-Growth Algorithm & Association Rule Mining Showcase[/bold cyan]\n"
            "[dim]High-Performance Frequent Itemset Mining without Candidate Generation[/dim]",
            box=box.DOUBLE,
            border_style="cyan"
        )
    )


def prompt_custom_transactions() -> List[List[str]]:
    """Interactive prompt for entering transactions one by one."""
    console.print("\n[bold yellow]Interactive Custom Transaction Input[/bold yellow]")
    console.print("[dim]Enter items separated by commas for each transaction.[/dim]")
    console.print("[dim]Press Enter on an empty line when finished.[/dim]\n")

    transactions: List[List[str]] = []
    tx_idx = 1
    while True:
        try:
            line = input(f"Transaction #{tx_idx} (e.g. Bread, Milk, Butter) > ").strip()
            if not line:
                if transactions:
                    break
                else:
                    console.print("[red]Please enter at least one transaction.[/red]")
                    continue
            items = [x for x in line.split(",") if x.strip()]
            cleaned = clean_transaction(items)
            if cleaned:
                transactions.append(cleaned)
                console.print(f"  [green]Added:[/green] {cleaned}")
                tx_idx += 1
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Finished input.[/yellow]")
            break

    return transactions


def run_showcase(
    transactions: List[List[str]],
    dataset_name: str,
    min_support: float = 0.25,
    min_confidence: float = 0.6,
    min_lift: float = 1.0,
    save_csv_path: Optional[str] = None
):
    console.print(f"\n[bold]Dataset:[/bold] [green]{dataset_name}[/green]")
    console.print(f"[bold]Total Transactions:[/bold] {len(transactions)}")
    console.print(
        f"[bold]Parameters:[/bold] min_support={min_support:.2f} "
        f"({int(min_support * len(transactions))} counts) | "
        f"min_confidence={min_confidence:.2f} | min_lift={min_lift:.2f}"
    )

    # 1. Preview Raw Transactions
    tx_table = Table(title="Sample Transactions Preview", box=box.ROUNDED, show_header=True)
    tx_table.add_column("TID", style="cyan", width=8)
    tx_table.add_column("Items in Basket", style="white")

    for idx, tx in enumerate(transactions[:8], start=1):
        tx_table.add_row(f"T{idx:03d}", ", ".join(tx))
    if len(transactions) > 8:
        tx_table.add_row("...", f"[dim]+ {len(transactions) - 8} more transactions[/dim]")
    console.print(tx_table)

    # 2. Fit FP-Growth Miner
    miner = FPGrowth(min_support=min_support)
    frequent_itemsets = miner.fit(transactions)
    trace = miner.trace

    # Step 1 & 2 Diagnostics
    step1_table = Table(title="Step 1 & 2: 1-Item Frequency & Pruning", box=box.SIMPLE_HEAD)
    step1_table.add_column("Item", style="bold")
    step1_table.add_column("Count", justify="right")
    step1_table.add_column("Support", justify="right")
    step1_table.add_column("Status", style="bold")

    for item, count in sorted(trace.raw_item_counts.items(), key=lambda x: -x[1]):
        supp = count / len(transactions)
        if count >= miner.effective_min_count:
            status = "[green]FREQUENT[/green]"
        else:
            status = "[dim red]PRUNED[/dim red]"
        step1_table.add_row(item, str(count), f"{supp:.1%}", status)
    console.print(step1_table)

    # Step 3 & 4: FP-Tree Display
    if miner.fp_tree:
        console.print(f"\n[bold magenta]Step 3 & 4: Constructed FP-Tree (Ordered Descending)[/bold magenta]")
        console.print(f"[dim]Header table order: {' -> '.join(trace.sorted_order)}[/dim]\n")
        
        def build_rich_tree(fp_node, rich_tree_node=None):
            if rich_tree_node is None:
                rich_tree_node = Tree("[bold cyan]ROOT (FP-Tree)[/bold cyan]")
                for child in fp_node.children.values():
                    build_rich_tree(child, rich_tree_node)
                return rich_tree_node
            branch = rich_tree_node.add(f"[green bold]{fp_node.item}[/green bold] [dim](count: {fp_node.count})[/dim]")
            for child in fp_node.children.values():
                build_rich_tree(child, branch)
            return rich_tree_node

        rich_tree = build_rich_tree(miner.fp_tree.root)
        console.print(rich_tree)

    # Step 5: Conditional Pattern Bases
    cpb_table = Table(title="Step 5: Mining Conditional Pattern Bases (Bottom-Up)", box=box.SIMPLE_HEAD)
    cpb_table.add_column("Item", style="cyan bold", width=16)
    cpb_table.add_column("Conditional Pattern Base (Prefix Paths : Count)", style="white")

    for item, paths in trace.conditional_pattern_bases.items():
        if paths:
            paths_str = ", ".join([f"({', '.join(p[0])} : {p[1]})" for p in paths])
        else:
            paths_str = "[dim](No non-root prefix paths)[/dim]"
        cpb_table.add_row(item, paths_str)
    console.print(cpb_table)

    # Frequent Itemsets Table
    fi_table = Table(title=f"Discovered Frequent Itemsets ({len(frequent_itemsets)} Found)", box=box.ROUNDED)
    fi_table.add_column("Length", justify="center", width=8)
    fi_table.add_column("Frequent Itemset", style="green bold")
    fi_table.add_column("Count", justify="right", width=10)
    fi_table.add_column("Support", justify="right", style="cyan", width=12)

    for itemset in frequent_itemsets:
        fi_table.add_row(
            str(len(itemset.items)),
            f"{{{', '.join(itemset.item_list)}}}",
            str(itemset.count),
            f"{itemset.support:.1%}"
        )
    console.print(fi_table)

    # Association Rules Mining
    rules = generate_association_rules(
        frequent_itemsets,
        min_confidence=min_confidence,
        min_lift=min_lift
    )

    ar_table = Table(
        title=f"Association Rules (Min Conf={min_confidence:.1%}, Min Lift={min_lift:.2f}) - {len(rules)} Rules",
        box=box.ROUNDED
    )
    ar_table.add_column("Rule (Antecedent => Consequent)", style="bold yellow")
    ar_table.add_column("Support", justify="right", style="cyan")
    ar_table.add_column("Confidence", justify="right", style="green bold")
    ar_table.add_column("Lift", justify="right", style="magenta bold")
    ar_table.add_column("Leverage", justify="right", style="dim")
    ar_table.add_column("Conviction", justify="right", style="dim")

    for rule in rules:
        conv_str = f"{rule.conviction:.2f}" if rule.conviction != float("inf") else "inf"
        ar_table.add_row(
            f"{{{rule.antecedent_str}}} => {{{rule.consequent_str}}}",
            f"{rule.support:.1%}",
            f"{rule.confidence:.1%}",
            f"{rule.lift:.2f}",
            f"{rule.leverage:+.3f}",
            conv_str
        )
    console.print(ar_table)

    # Business Insights Summary
    if rules:
        top_rule = rules[0]
        console.print(
            Panel(
                f"[bold green]Top Association Pattern Highlight:[/bold green]\n"
                f"Customers who purchase [yellow]{{{top_rule.antecedent_str}}}[/yellow] have a "
                f"[bold cyan]{top_rule.confidence:.1%}[/bold cyan] probability of also buying "
                f"[bold yellow]{{{top_rule.consequent_str}}}[/bold yellow] "
                f"(Lift: [magenta]{top_rule.lift:.2f}x[/magenta] higher than random chance).\n\n"
                f"[dim]Strategic Recommendation: Bundle these items or position them along the same navigation aisle.[/dim]",
                box=box.DOUBLE,
                border_style="green"
            )
        )

    # Optional CSV Export
    if save_csv_path:
        csv_data = export_transactions_to_csv(transactions)
        Path(save_csv_path).write_text(csv_data, encoding="utf-8")
        console.print(f"[green]Saved {len(transactions)} transactions to:[/green] {save_csv_path}")


def main():
    parser = argparse.ArgumentParser(description="FP-Growth Algorithm Showcase CLI")
    parser.add_argument(
        "--dataset",
        choices=["grocery", "electronics", "streaming", "cafe"],
        default="grocery",
        help="Use a pre-configured benchmark dataset (default: grocery)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to custom CSV file with transactions"
    )
    parser.add_argument(
        "--custom-text",
        type=str,
        default=None,
        help="Custom multiline transactions passed as a string"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively add custom transactions in terminal"
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.25,
        help="Minimum support fraction (default: 0.25)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence threshold (default: 0.60)"
    )
    parser.add_argument(
        "--min-lift",
        type=float,
        default=1.0,
        help="Minimum lift threshold (default: 1.0)"
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Export loaded transactions to a CSV file"
    )

    args = parser.parse_args()
    display_welcome_banner()

    transactions: List[List[str]] = []
    dataset_name = ""

    if args.interactive:
        transactions = prompt_custom_transactions()
        dataset_name = "Interactive Custom Input"
    elif args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            console.print(f"[red]Error: File not found at '{args.csv}'[/red]")
            sys.exit(1)
        transactions, detected_fmt = parse_csv_file(csv_path)
        dataset_name = f"CSV File '{csv_path.name}' ({detected_fmt})"
    elif args.custom_text:
        transactions = parse_multiline_text(args.custom_text)
        dataset_name = "Custom String Input"
    else:
        dataset_info = BENCHMARK_DATASETS[args.dataset]
        transactions = dataset_info["transactions"]  # type: ignore
        dataset_name = str(dataset_info["name"])

    if not transactions:
        console.print("[red]No transactions available to mine. Exiting.[/red]")
        sys.exit(1)

    run_showcase(
        transactions=transactions,
        dataset_name=dataset_name,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        save_csv_path=args.export_csv
    )


if __name__ == "__main__":
    main()
