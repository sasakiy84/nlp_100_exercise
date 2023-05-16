
from copy import copy
from re import M
from typing import Any, TypedDict
from xml.etree.ElementTree import Element

from morph import Morph
from utils import parse_token_to_morph


class ChunkInitArgs(TypedDict):
    id: int
    morphs: list[Morph]
    dst: int
    srcs: list[int]


class Chunk(object):
    def __init__(self, init_args: ChunkInitArgs) -> None:
        self.id = init_args['id']
        self.morphs = init_args['morphs']
        self.dst = init_args['dst']
        self.srcs = init_args['srcs']

    def to_text(self, exclude_pos: list[str] = []) -> str:
        text = ""
        for m in self.morphs:
            if m.pos in exclude_pos:
                continue
            text += m.surface
        return text


class Sentece(object):
    def __init__(self, sentence: Element) -> None:
        # register Chunks except for srcs
        chunk_list: list[Chunk] = []
        for chunk in sentence:
            morph_list: list[Morph] = []
            for token in chunk:
                morph_list.append(parse_token_to_morph(token))

            chunk_list.append(Chunk({
                "morphs": morph_list,
                "id": int(chunk.attrib["id"]),
                "dst": int(chunk.attrib["link"]),
                "srcs": []
            }))

        # register srcs
        for chunk in chunk_list:
            dst = chunk.dst
            if dst == -1:
                continue

            for dst_chunk_canditate in chunk_list:
                if dst_chunk_canditate.id == dst:
                    dst_chunk_canditate.srcs.append(chunk.id)
                    break

        self.chunks = chunk_list

        self.tree = self.construct_tree()

    def find_chunk_by_id(self, id: int) -> Chunk | None:
        for chunk in self.chunks:
            if chunk.id == id:
                return chunk

    def find_chunk_by_dst_id(self, id: int) -> Chunk | None:
        for chunk in self.chunks:
            if chunk.dst == id:
                return copy(chunk)

    def dependency_as_text(self, exclude_pos: list[str] = [], divider=" -> ", required_pos_in_src: list[str] = [], required_pos_in_dst: list[str] = []) -> str:
        rows: list[str] = []

        for chunk in self.chunks:
            meet_requirement_of_src = False
            meet_requirement_of_dst = False

            if len(required_pos_in_src) != 0:
                for m in chunk.morphs:
                    if m.pos in required_pos_in_src:
                        meet_requirement_of_src = True

                if meet_requirement_of_src == False:
                    continue

            dst_chunk = self.find_chunk_by_id(chunk.dst)

            if len(required_pos_in_dst) != 0:
                if dst_chunk == None:
                    continue

                for m in dst_chunk.morphs:
                    if m.pos in required_pos_in_dst:
                        meet_requirement_of_dst = True

                if meet_requirement_of_dst == False:
                    continue

            if dst_chunk == None:
                rows.append(
                    f"{chunk.to_text(exclude_pos=exclude_pos)}{divider}ROOT")
            else:
                rows.append(
                    f"{chunk.to_text(exclude_pos=exclude_pos)}{divider}{dst_chunk.to_text(exclude_pos=exclude_pos)}")

        return "\n".join(rows)

    def construct_tree(self) -> Any:
        root = self.find_chunk_by_dst_id(-1)

        def construct_tree_partial(chunk: Chunk):
            children = []
            for src_id in chunk.srcs:
                children.append(self.find_chunk_by_id(src_id))

            constructed_children = []
            for child in children:
                constructed_children.append(construct_tree_partial(child))

            return {
                "children": constructed_children,
                "id": chunk.id,
                "morphs": chunk.morphs,
                "text": chunk.to_text(),
                "dst": chunk.dst,
                "srcs": chunk.srcs
            }

        return construct_tree_partial(root)

    def tree_as_text(self, indent_str="    ", indent_last=" |--") -> str:
        rows: list[str] = []

        def walk_tree(node, depth):
            indent = indent_str * (depth - 1) + \
                indent_last if depth > 0 else ""
            rows.append(f"{indent}{node['text']}")
            for child in node["children"]:
                walk_tree(child, depth+1)

        walk_tree(self.tree, 0)
        return "\n".join(rows)
