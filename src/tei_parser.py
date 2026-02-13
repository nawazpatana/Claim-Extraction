from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

from lxml import etree

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _norm(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _first_text(nodes) -> Optional[str]:
    if not nodes:
        return None
    n = nodes[0]
    if isinstance(n, etree._Element):
        return _norm("".join(n.itertext()))
    return _norm(str(n))


def _all_text(nodes) -> List[str]:
    out = []
    for n in nodes:
        if isinstance(n, etree._Element):
            t = _norm("".join(n.itertext()))
        else:
            t = _norm(str(n))
        if t:
            out.append(t)
    return out


def _extract_year(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"(19|20)\d{2}", s)
    return int(m.group(0)) if m else None


def _parse_xml(path: Union[str, Path]) -> etree._ElementTree:
    parser = etree.XMLParser(
        recover=True,
        remove_comments=True,
        resolve_entities=False,
        huge_tree=True,
    )
    return etree.parse(str(path), parser)


@dataclass
class TeiMetadata:
    doi: Optional[str] = None
    year: Optional[int] = None
    title: Optional[str] = None
    abstract: Optional[str] = None

    # optional extras (not required but useful)
    authors: Optional[List[str]] = None
    venue: Optional[str] = None
    source_file: Optional[str] = None


def parse_tei_metadata(tei_path: Union[str, Path]) -> Dict[str, Any]:
    tei_path = Path(tei_path)
    tree = _parse_xml(tei_path)
    root = tree.getroot()

    doi = _first_text(root.xpath("//tei:teiHeader//tei:idno[@type='DOI']", namespaces=TEI_NS))
    if doi:
        doi = doi.replace("doi:", "").strip()
        doi = re.sub(r"\s+", "", doi)

    title = (
        _first_text(root.xpath(
            "//tei:teiHeader//tei:sourceDesc//tei:biblStruct//tei:analytic//tei:title[@type='main']",
            namespaces=TEI_NS
        ))
        or _first_text(root.xpath("//tei:teiHeader//tei:fileDesc//tei:titleStmt//tei:title", namespaces=TEI_NS))
        or _first_text(root.xpath("//tei:text//tei:front//tei:docTitle//tei:titlePart", namespaces=TEI_NS))
    )

    date_when = _first_text(root.xpath(
        "//tei:teiHeader//tei:sourceDesc//tei:biblStruct//tei:monogr//tei:imprint//tei:date/@when",
        namespaces=TEI_NS
    ))
    date_text = _first_text(root.xpath(
        "//tei:teiHeader//tei:sourceDesc//tei:biblStruct//tei:monogr//tei:imprint//tei:date",
        namespaces=TEI_NS
    ))
    year = _extract_year(date_when or date_text)

    abstract = (
        _first_text(root.xpath("//tei:teiHeader//tei:profileDesc//tei:abstract", namespaces=TEI_NS))
        or _first_text(root.xpath("//tei:text//tei:front//tei:div[@type='abstract']", namespaces=TEI_NS))
    )
    if not abstract:
        abstract = _first_text(root.xpath(
            "(//tei:teiHeader//tei:profileDesc//tei:abstract//tei:p"
            " | //tei:text//tei:front//tei:div[@type='abstract']//tei:p)[1]",
            namespaces=TEI_NS
        ))

    # authors (optional)
    author_nodes = root.xpath(
        "//tei:teiHeader//tei:sourceDesc//tei:biblStruct//tei:analytic//tei:author//tei:persName"
        " | //tei:teiHeader//tei:fileDesc//tei:titleStmt//tei:author//tei:persName"
        " | //tei:teiHeader//tei:fileDesc//tei:titleStmt//tei:author",
        namespaces=TEI_NS
    )
    authors = []
    for a in author_nodes:
        t = _norm("".join(a.itertext())) if isinstance(a, etree._Element) else _norm(str(a))
        if t and t not in authors:
            authors.append(t)
    if not authors:
        authors = None

    venue = _first_text(root.xpath(
        "//tei:teiHeader//tei:sourceDesc//tei:biblStruct//tei:monogr//tei:title[@type='main']"
        " | //tei:teiHeader//tei:sourceDesc//tei:biblStruct//tei:monogr//tei:title",
        namespaces=TEI_NS
    ))

    meta = TeiMetadata(
        doi=doi,
        year=year,
        title=title,
        abstract=abstract,
        authors=authors,
        venue=venue,
        source_file=tei_path.name,
    )
    return asdict(meta)
