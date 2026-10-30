from __future__ import annotations
import csv
import io
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

BENCHMARK_DATASETS: Dict[str, Dict[str, Union[str, List[List[str]]]]] = {
    "grocery": {
        "name": "Supermarket Grocery Market Basket",
        "description": "Classic retail transactions including bread, milk, diapers, beer, and snacks.",
        "transactions": [
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
            ["Diapers", "Beer", "Chips", "Cola"],
            ["Milk", "Bread", "Diapers", "Beer", "Butter"],
            ["Milk", "Eggs", "Butter"],
            ["Bread", "Chips", "Cola"],
            ["Milk", "Bread", "Butter", "Cola"],
        ]
    },
    "electronics": {
        "name": "Tech & PC Gaming Store",
        "description": "E-commerce orders of PC hardware, gaming peripherals, and home office accessories.",
        "transactions": [
            ["Gaming PC", "Mechanical Keyboard", "Gaming Mouse", "Ultrawide Monitor"],
            ["Mechanical Keyboard", "Gaming Mouse", "Mousepad"],
            ["Laptop", "USB-C Dock", "Wireless Mouse", "HDMI Cable"],
            ["Gaming PC", "Ultrawide Monitor", "Headset", "Mechanical Keyboard"],
            ["Wireless Mouse", "Laptop", "USB-C Dock", "Webcam"],
            ["Mechanical Keyboard", "Gaming Mouse", "Headset", "Mousepad"],
            ["Gaming PC", "Mechanical Keyboard", "Gaming Mouse"],
            ["Laptop", "Wireless Mouse", "Laptop Stand", "USB-C Dock"],
            ["Ultrawide Monitor", "Monitor Arm", "HDMI Cable"],
            ["Mechanical Keyboard", "Wrist Rest", "Gaming Mouse"],
            ["Gaming PC", "Ultrawide Monitor", "Mechanical Keyboard", "Gaming Mouse", "Headset"],
            ["Laptop", "USB-C Dock", "Webcam", "Headset"],
            ["Mechanical Keyboard", "Gaming Mouse", "Mousepad", "Headset"],
            ["Laptop Stand", "Wireless Mouse", "Laptop"],
            ["Gaming PC", "Gaming Mouse", "Headset"],
        ]
    },
    "streaming": {
        "name": "Movie & Show Streaming Watchlists",
        "description": "Viewer co-consumption patterns across genres and formats.",
        "transactions": [
            ["Sci-Fi", "Action", "Cyberpunk"],
            ["Action", "Thriller", "Crime"],
            ["Sci-Fi", "Cyberpunk", "Animation"],
            ["Romantic Comedy", "Drama", "Slice of Life"],
            ["Sci-Fi", "Action", "Thriller", "Cyberpunk"],
            ["Horror", "Thriller", "Mystery"],
            ["Action", "Adventure", "Sci-Fi"],
            ["Drama", "Documentary", "Historical"],
            ["Sci-Fi", "Space Opera", "Action"],
            ["Comedy", "Animation", "Family"],
            ["Action", "Thriller", "Mystery", "Crime"],
            ["Sci-Fi", "Cyberpunk", "Action", "Animation"],
            ["Romantic Comedy", "Drama", "Romance"],
            ["Horror", "Mystery", "Thriller"],
            ["Sci-Fi", "Action", "Thriller"],
        ]
    },
    "cafe": {
        "name": "Artisan Bakery & Cafe",
        "description": "Morning breakfast and afternoon coffee orders.",
        "transactions": [
            ["Espresso", "Croissant"],
            ["Cappuccino", "Blueberry Muffin", "Orange Juice"],
            ["Latte", "Avocado Toast"],
            ["Espresso", "Croissant", "Chocolate Cookie"],
            ["Iced Americano", "Bagel with Cream Cheese"],
            ["Cappuccino", "Croissant", "Espresso"],
            ["Latte", "Blueberry Muffin"],
            ["Espresso", "Avocado Toast", "Orange Juice"],
            ["Chai Latte", "Cinnamon Roll"],
            ["Iced Americano", "Bagel with Cream Cheese", "Cookie"],
            ["Cappuccino", "Croissant"],
            ["Espresso", "Croissant", "Bottled Water"],
            ["Latte", "Croissant", "Avocado Toast"],
            ["Cappuccino", "Blueberry Muffin"],
            ["Espresso", "Iced Americano", "Croissant"],
        ]
    }
}


def clean_item(item: str) -> str:
    
    return item.strip().strip("'\"")


def clean_transaction(items: List[str]) -> List[str]:
    
    cleaned = []
    seen = set()
    for raw in items:
        cleaned_item = clean_item(raw)
        if cleaned_item and cleaned_item not in seen:
            seen.add(cleaned_item)
            cleaned.append(cleaned_item)
    return cleaned


def parse_multiline_text(text: str) -> List[List[str]]:
  
    transactions: List[List[str]] = []
    lines = text.strip().splitlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        delimiter = ";" if ";" in line and "," not in line else ","
        raw_items = [item for item in line.split(delimiter) if item.strip()]
        cleaned = clean_transaction(raw_items)
        if cleaned:
            transactions.append(cleaned)

    return transactions


def parse_csv_file(file_content_or_path: Union[str, Path, io.StringIO, bytes]) -> Tuple[List[List[str]], str]:
  
    if isinstance(file_content_or_path, bytes):
        text_stream = io.StringIO(file_content_or_path.decode("utf-8", errors="replace"))
    elif isinstance(file_content_or_path, (str, Path)) and "\n" not in str(file_content_or_path) and Path(file_content_or_path).is_file():
        with open(file_content_or_path, "r", encoding="utf-8", errors="replace") as f:
            text_stream = io.StringIO(f.read())
    elif isinstance(file_content_or_path, str):
        text_stream = io.StringIO(file_content_or_path)
    else:
        text_stream = file_content_or_path

    reader = csv.reader(text_stream)
    rows = [row for row in reader if row and any(cell.strip() for cell in row)]

    if not rows:
        return [], "Empty CSV"

    header = [cell.strip().lower() for cell in rows[0]]

    basket_col_candidates = ["items", "products", "basket", "baskets", "item_list", "itemsets"]
    basket_col_idx = None
    for cand in basket_col_candidates:
        if cand in header:
            basket_col_idx = header.index(cand)
            break

    if basket_col_idx is not None:
        transactions = []
        for row in rows[1:]:
            if len(row) > basket_col_idx:
                raw_cell = row[basket_col_idx]
                delim = "|" if "|" in raw_cell else (";" if ";" in raw_cell else ",")
                items = [clean_item(x) for x in raw_cell.split(delim)]
                cleaned = clean_transaction(items)
                if cleaned:
                    transactions.append(cleaned)
        return transactions, f"Basket format (Extracted from column '{rows[0][basket_col_idx]}')"

    tx_id_col = None
    item_col = None

    for idx, col in enumerate(header):
        if any(term in col for term in ["transaction_id", "order_id", "invoice_no", "invoice", "tid", "basket_id", "cart_id"]):
            tx_id_col = idx
        elif any(term in col for term in ["item_id", "product_id", "product_name", "item_name", "sku", "article", "item", "product"]):
            item_col = idx

    if tx_id_col is not None and item_col is not None and len(header) <= 6:
        grouped: Dict[str, List[str]] = {}
        for row in rows[1:]:
            if len(row) > max(tx_id_col, item_col):
                tid = clean_item(row[tx_id_col])
                raw_item = clean_item(row[item_col])
                if tid and raw_item:        
                    if "," in raw_item:
                        items = [clean_item(x) for x in raw_item.split(",")]
                    else:
                        items = [raw_item]
                    if tid not in grouped:
                        grouped[tid] = []
                    grouped[tid].extend(items)
        transactions = [clean_transaction(items) for items in grouped.values() if items]
        return transactions, f"Long format (Grouped by column '{rows[0][tx_id_col]}')"

    if len(rows) > 1 and len(header) >= 2:
        sample_rows = rows[1:min(6, len(rows))]
        matrix_hits = 0
        total_cells = 0
        for r in sample_rows:
            for cell in r:
                total_cells += 1
                val = cell.strip().lower()
                if val in ["0", "1", "true", "false", "yes", "no", "t", "f", ""]:
                    matrix_hits += 1

        if total_cells > 0 and (matrix_hits / total_cells) > 0.8:
            items_headers = [clean_item(c) for c in rows[0]]
            transactions = []
            for row in rows[1:]:
                tx = []
                for idx, cell in enumerate(row):
                    if idx < len(items_headers):
                        val = cell.strip().lower()
                        if val in ["1", "true", "yes", "t"]:
                            tx.append(items_headers[idx])
                if tx:
                    transactions.append(clean_transaction(tx))
            return transactions, f"Binary Matrix format ({len(items_headers)} item columns)"

    start_idx = 0
    if all(any(token in cell.lower() for token in ["item", "col", "var"]) for cell in rows[0]):
        start_idx = 1

    transactions = []
    for row in rows[start_idx:]:
        cleaned = clean_transaction(row)
        if cleaned:
            transactions.append(cleaned)

    return transactions, "Row-per-basket format"


def export_transactions_to_csv(transactions: List[List[str]], format_type: str = "basket") -> str:
    
    output = io.StringIO()
    writer = csv.writer(output)

    if format_type == "long":
        writer.writerow(["transaction_id", "item"])
        for tid, tx in enumerate(transactions, start=1):
            for item in tx:
                writer.writerow([f"T{tid:04d}", item])
    else:
        writer.writerow(["transaction_id", "items"])
        for tid, tx in enumerate(transactions, start=1):
            writer.writerow([f"T{tid:04d}", ", ".join(tx)])

    return output.getvalue()
