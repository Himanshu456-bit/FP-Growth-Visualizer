from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set, Any


class FPNode:

    def __init__(self, item: Optional[str], count: int = 1, parent: Optional[FPNode] = None):
        self.item: Optional[str] = item
        self.count: int = count
        self.parent: Optional[FPNode] = parent
        self.children: Dict[str, FPNode] = {}
        self.node_link: Optional[FPNode] = None  

    @property
    def is_root(self) -> bool:
        return self.item is None

    def increment(self, count: int = 1) -> None:
        self.count += count

    def get_path_to_root(self) -> List[str]:
        
        path: List[str] = []
        curr = self.parent
        while curr and not curr.is_root:
            if curr.item is not None:
                path.append(curr.item)
            curr = curr.parent
        path.reverse()
        return path

    def __repr__(self) -> str:
        return f"FPNode({self.item}: count={self.count})"


class FPTree:

    def __init__(self):
        self.root: FPNode = FPNode(item=None, count=0, parent=None)
        self.header_table: Dict[str, List[Any]] = {}

    def add_header_item(self, item: str, count: int) -> None:
        if item not in self.header_table:
            self.header_table[item] = [count, None]

    def insert_transaction(self, transaction: List[str], count: int = 1) -> None:
        curr = self.root
        for item in transaction:
            if item in curr.children:
                curr.children[item].increment(count)
            else:
                new_node = FPNode(item=item, count=count, parent=curr)
                curr.children[item] = new_node

                if item in self.header_table:
                    if self.header_table[item][1] is None:
                        self.header_table[item][1] = new_node
                    else:
                        chain = self.header_table[item][1]
                        while chain.node_link is not None:
                            chain = chain.node_link
                        chain.node_link = new_node

            curr = curr.children[item]

    def get_nodes_for_item(self, item: str) -> List[FPNode]:        
        nodes = []
        if item in self.header_table:
            curr = self.header_table[item][1]
            while curr is not None:
                nodes.append(curr)
                curr = curr.node_link
        return nodes

    def format_tree(
        self,
        node: Optional[FPNode] = None,
        prefix: str = "",
        is_last: bool = True,
        ascii_only: bool = False
    ) -> List[str]:

        lines: List[str] = []
        if node is None:
            node = self.root
            lines.append("ROOT")
            child_keys = list(node.children.keys())
            for idx, child_key in enumerate(child_keys):
                is_child_last = (idx == len(child_keys) - 1)
                lines.extend(self.format_tree(node.children[child_key], "", is_child_last, ascii_only=ascii_only))
            return lines

        if ascii_only:
            branch = "\\-- " if is_last else "+-- "
            pipe = "    " if is_last else "|   "
        else:
            branch = "└── " if is_last else "├── "
            pipe = "    " if is_last else "│   "

        lines.append(f"{prefix}{branch}{node.item} (count: {node.count})")

        new_prefix = prefix + pipe
        child_keys = list(node.children.keys())
        for idx, child_key in enumerate(child_keys):
            is_child_last = (idx == len(child_keys) - 1)
            lines.extend(self.format_tree(node.children[child_key], new_prefix, is_child_last, ascii_only=ascii_only))

        return lines

    def print_tree(self) -> None:
        print("\n".join(self.format_tree()))
