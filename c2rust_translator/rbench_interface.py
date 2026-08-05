from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import re

from tree_sitter import Language, Node, Parser
import tree_sitter_rust as ts_rust


RUST = Language(ts_rust.language())
PARSER = Parser(RUST)


@dataclass(frozen=True)
class Callable:
    module: str
    kind: str
    name: str
    type_name: Optional[str]
    file_name: str
    signature: str
    signature_key: str
    param_count: int
    stub: str


def _text(source: bytes, node: Node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _first_child(node: Node, kind: str) -> Optional[Node]:
    for child in node.children:
        if child.type == kind:
            return child
    return None


def _ancestor(node: Node, kind: str) -> Optional[Node]:
    current = node.parent
    while current is not None:
        if current.type == kind:
            return current
        current = current.parent
    return None


def _function_name(node: Node, source: bytes) -> Optional[str]:
    name = node.child_by_field_name("name")
    if name is not None:
        return _text(source, name).strip()
    for child in node.children:
        if child.type == "identifier":
            return _text(source, child).strip()
    return None


def _impl_type_name(node: Node, source: bytes) -> Optional[str]:
    type_node = node.child_by_field_name("type")
    if type_node is None:
        return None

    stack = [type_node]
    while stack:
        current = stack.pop()
        if current.type == "type_identifier":
            return _text(source, current).strip()
        stack.extend(reversed(current.children))
    return None


def _impl_trait_name(node: Node, source: bytes) -> Optional[str]:
    trait_node = node.child_by_field_name("trait")
    if trait_node is None:
        return None
    return _text(source, trait_node).strip()


def _signature(node: Node, source: bytes) -> str:
    block = _first_child(node, "block")
    if block is None:
        full = _text(source, node).strip()
        return full.split("{", 1)[0].strip()
    return source[node.start_byte:block.start_byte].decode("utf-8", errors="replace").strip()


def _signature_key(signature: str) -> str:
    return re.sub(r"\s+", "", signature or "")


def _param_count(node: Node) -> int:
    params = node.child_by_field_name("parameters")
    if params is None:
        return 0

    count = 0
    for child in params.children:
        if child.type == "parameter":
            count += 1
    return count


def _is_stub(text: str) -> bool:
    return "unimplemented!" in text or "todo!" in text


def _is_public(signature: str) -> bool:
    prefix = signature.split("fn", 1)[0]
    return "pub" in prefix or signature.strip().startswith("pub ")


class InterfaceIndex:
    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.interfaces_dir = self.project_dir / "src" / "interfaces"
        self._items: list[Callable] = []
        self._public_items: list[Callable] = []

    def files(self) -> Iterable[Path]:
        if not self.interfaces_dir.exists():
            return []
        return sorted(self.interfaces_dir.glob("*.rs"))

    def build(self) -> "InterfaceIndex":
        self._items = []
        self._public_items = []

        for path in self.files():
            module = path.stem
            source = path.read_bytes()
            root = PARSER.parse(source).root_node
            stack = [root]

            while stack:
                node = stack.pop()
                stack.extend(reversed(node.children))
                if node.type != "function_item":
                    continue

                name = _function_name(node, source)
                if not name:
                    continue

                signature = _signature(node, source)
                impl = _ancestor(node, "impl_item")
                body = _text(source, node).strip()
                is_public = _is_public(signature)
                is_trait_stub = impl is not None and _impl_trait_name(impl, source) is not None and _is_stub(body)
                if not is_public and not is_trait_stub:
                    continue

                callable_kind = "method" if impl is not None else "fn"
                type_name = _impl_type_name(impl, source) if impl is not None else None
                item = Callable(
                    module=module,
                    kind=callable_kind,
                    name=name,
                    type_name=type_name,
                    file_name=path.name,
                    signature=signature,
                    signature_key=_signature_key(signature),
                    param_count=_param_count(node),
                    stub=body,
                )
                self._public_items.append(item)
                if (is_public or is_trait_stub) and _is_stub(body):
                    self._items.append(item)

        return self

    def list_unimplemented(self) -> list[Callable]:
        return list(self._items)

    def list_public(self) -> list[Callable]:
        return list(self._public_items)
