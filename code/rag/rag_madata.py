from __future__ import annotations
import re, html, unicodedata
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import json
import time
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm
try:
    import faiss  # faiss-cpu
    FAISS_OK = True
except Exception:
    FAISS_OK = False

# ---------------------------
# 0) OAI-PMH Harvest utilities (Sickle)
# ---------------------------

try:
    from sickle import Sickle
    HAVE_SICKLE = True
except Exception:
    HAVE_SICKLE = False


def require_sickle() -> None:
    if not HAVE_SICKLE:
        raise RuntimeError("sickle is not installed. Install with: pip install sickle")


def harvest_oai_dc(
    base_url: str = "https://madata.bib.uni-mannheim.de/cgi/oai2",
    metadata_prefix: str = "oai_dc",
    set_spec: Optional[str] = None,
    max_records: Optional[int] = None,
    sleep_secs: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Harvest records from an OAI-PMH endpoint using Sickle and convert each record's XML to a dict.

    - base_url: OAI-PMH endpoint
    - metadata_prefix: e.g. 'oai_dc'
    - set_spec: optional OAI set specifier
    - max_records: limit number of records for quick testing
    - sleep_secs: polite delay between pages
    """
    require_sickle()
    sickle = Sickle(base_url)
    params: Dict[str, Any] = {"metadataPrefix": metadata_prefix}
    if set_spec:
        params["set"] = set_spec

    records_iter = sickle.ListRecords(**params)
    harvested: List[Dict[str, Any]] = []

    for record in tqdm(records_iter, desc="Harvesting OAI-PMH records"):
        try:
            root = ET.fromstring(record.raw)
            rec_dict = xml_to_dict(root)
            harvested.append(rec_dict)
        except Exception as e:
            # Skip bad record, continue harvest
            print(f"[warn] Error parsing record: {e}")
            continue
        if max_records is not None and len(harvested) >= max_records:
            break
        if sleep_secs > 0:
            time.sleep(sleep_secs)

    return harvested


def save_results_json(results: List[Dict[str, Any]], path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


def load_results_json(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------
# 1) Utils
# ---------------------------

def strip_html(s: Optional[str]) -> str:
    if not s:
        return ""
    # remove tags (cheap), unescape entities, normalize ws
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = " ".join(s.split())
    return s.strip()


def to_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(y) for y in x]
    return [str(x)]


def _split_people_field(val: str) -> List[str]:
    # Split common multi-name delimiters without breaking "Last, First" forms
    parts = re.split(r"\s*;\s*|\s*\|\s*|\s+and\s+", val)
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def norm_whitespace(s: str) -> str:
    return " ".join(str(s).split())


def year_from_date(d: Optional[str]) -> Optional[int]:
    if not d:
        return None
    m = re.search(r"(\d{4})", d)
    return int(m.group(1)) if m else None


DOI_RX = re.compile(r"\b10\.\d{4,9}/\S+\b", re.I)


def extract_doi(*candidates: List[str]) -> Optional[str]:
    for seq in candidates:
        for item in to_list(seq):
            # plain DOI or URL containing DOI
            m = DOI_RX.search(item)
            if m:
                # strip trailing punctuation
                doi = m.group(0).rstrip(").,;]")
                return doi
    return None


def pick_landing_url(*candidates: List[str]) -> Optional[str]:
    for seq in candidates:
        for item in to_list(seq):
            if isinstance(item, str) and item.startswith("http"):
                # prefer madata landing pages
                if "madata.bib.uni-mannheim.de" in item and re.search(r"/\d+/?$", item):
                    return item
    # fallback: first http link
    for seq in candidates:
        for item in to_list(seq):
            if isinstance(item, str) and item.startswith("http"):
                return item
    return None


def safe_get(d: Dict, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def get_tag_without_namespace(elem):
    return elem.tag.split('}', 1)[-1] if '}' in elem.tag else elem.tag


def xml_to_dict(element):
    if not isinstance(element, ET.Element):
        return element
    if len(element) == 0 and element.text:
        return element.text.strip()
    result = {}
    for child in element:
        child_tag = get_tag_without_namespace(child)
        child_dict = xml_to_dict(child)
        if child_tag not in result:
            result[child_tag] = child_dict
        else:
            if not isinstance(result[child_tag], list):
                result[child_tag] = [result[child_tag]]
            result[child_tag].append(child_dict)
    return result


def normalize_name(name: str) -> str:
    # lightweight display normalization (don't over-normalize)
    name = unicodedata.normalize("NFKC", name).strip()
    return re.sub(r"\s+", " ", name)


# ---------------------------
# 2) Normalized record model
# ---------------------------

@dataclass
class Record:
    id: str
    repo: str
    title: str
    creators: List[str]
    subjects: List[str]
    description: str
    year: Optional[int]
    types: List[str]
    languages: List[str]
    formats: List[str]
    rights: List[str]
    keywords: List[str]  # we’ll mirror subjects here (DC combines them)
    doi: Optional[str]
    landing_url: Optional[str]
    identifiers: List[str]
    relations: List[str]
    extra: Dict[str, Any]

    @property
    def citation_id(self) -> str:
        # stable handle for citations
        return self.doi or self.landing_url or self.id

    def index_text(self) -> str:
        # High-signal, compact representation for retrieval
        pieces = [
            self.title,
            f"Authors: {', '.join(self.creators)}" if self.creators else "",
            f"Year: {self.year}" if self.year else "",
            f"Subjects: {', '.join(self.subjects)}" if self.subjects else "",
            f"Keywords: {', '.join(self.keywords)}" if self.keywords else "",
            strip_html(self.description),
        ]
        return norm_whitespace("\n".join([p for p in pieces if p]))


def _ensure_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def normalize_oai_dc(results: List[Dict[str, Any]], repo_name: str = "madata") -> List[Record]:
    out: List[Record] = []
    for rec in results:
        # Expect OAI-PMH structure: record -> header, metadata -> dc
        header = _ensure_dict(rec.get("header"))
        metadata = _ensure_dict(rec.get("metadata"))
        dc = _ensure_dict(metadata.get("dc")) or {}

        header_id = header.get("identifier") or f"{repo_name}:unknown"

        title = dc.get("title") or _ensure_dict(metadata.get("dc")).get("ubma_additional_title") or ""
        title = strip_html(str(title))

        # authors: include creators and contributors
        creators_raw = to_list(dc.get("creator"))
        contributors_raw = to_list(dc.get("contributor")) + to_list(dc.get("ubma_author")) + to_list(dc.get("ubma_authors"))
        creators_all = []
        for item in creators_raw + contributors_raw:
            creators_all.extend(_split_people_field(str(item)))
        creators = []
        seen = set()
        for c in creators_all:
            cn = normalize_name(str(c))
            if cn and cn.lower() not in seen:
                creators.append(cn)
                seen.add(cn.lower())

        subjects = [s.strip() for s in to_list(dc.get("subject")) if s and str(s).strip()]
        keywords = list(subjects)

        description = strip_html(dc.get("description") or "")
        year = year_from_date(dc.get("date"))
        types = to_list(dc.get("type"))
        languages = to_list(dc.get("language") or dc.get("ubma_language"))
        formats = to_list(dc.get("format"))
        rights = to_list(dc.get("rights"))

        identifiers = to_list(dc.get("identifier"))
        relations = to_list(dc.get("relation"))

        doi = extract_doi(dc.get("id_number"), identifiers, relations)
        landing = pick_landing_url(relations, identifiers)

        extra: Dict[str, Any] = {}
        for k in list(dc.keys()):
            if k.startswith("ubma_") or k in ("id_number", "ubma_publications", "ubma_project"):
                extra[k] = dc[k]

        out.append(Record(
            id=str(header_id),
            repo=repo_name,
            title=title,
            creators=creators,
            subjects=subjects,
            description=description,
            year=year,
            types=types,
            languages=languages,
            formats=formats,
            rights=rights,
            keywords=keywords,
            doi=doi,
            landing_url=landing,
            identifiers=identifiers,
            relations=relations,
            extra=extra
        ))
    return out



def _flatten_to_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    out: List[str] = []
    if isinstance(value, (list, tuple, set)):
        for item in value:
            out.extend(_flatten_to_list(item))
        return out
    if isinstance(value, dict):
        for item in value.values():
            out.extend(_flatten_to_list(item))
        return out
    return [str(value)]


def _clean_string_list(value) -> List[str]:
    cleaned: List[str] = []
    for item in _flatten_to_list(value):
        s = strip_html(str(item)).strip()
        if s:
            cleaned.append(s)
    return cleaned


def _as_iterable_preserve(value) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _first_string(value: Any) -> str:
    for item in _flatten_to_list(value):
        if item is None:
            continue
        if isinstance(item, str):
            s = item.strip()
            if s:
                return s
        else:
            s = str(item).strip()
            if s:
                return s
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _extract_people(block) -> List[str]:
    if block is None:
        return []
    entries: List[Any] = []
    if isinstance(block, dict):
        for key in ("creator", "contributor", "person", "name"):
            if key in block:
                entries = _as_iterable_preserve(block.get(key))
                break
        else:
            entries = _as_iterable_preserve(block)
    else:
        entries = _as_iterable_preserve(block)

    names: List[str] = []
    for entry in entries:
        name = ""
        if isinstance(entry, dict):
            name = _first_string(entry.get("creatorName")) or _first_string(entry.get("contributorName")) or _first_string(entry.get("name"))
            if not name:
                given = _first_string(entry.get("givenName") or entry.get("firstName"))
                family = _first_string(entry.get("familyName") or entry.get("lastName"))
                if family and given:
                    name = f"{family}, {given}"
                elif family or given:
                    name = family or given
            if not name and "value" in entry:
                name = _first_string(entry.get("value"))
            if not name and len(entry) == 1:
                name = _first_string(next(iter(entry.values())))
        else:
            name = _first_string(entry)
        name = normalize_name(name) if name else ""
        if name:
            names.append(name)
    return names


def normalize_openaire(results: List[Dict[str, Any]], repo_name: str = "madoc") -> List[Record]:
    out: List[Record] = []
    for idx, rec in enumerate(results):
        header = _ensure_dict(rec.get("header"))
        metadata = _ensure_dict(rec.get("metadata"))
        resource = _ensure_dict(metadata.get("resource"))
        if not resource:
            continue

        header_id = header.get("identifier") or resource.get("identifier") or f"{repo_name}:{idx}"

        titles_field = resource.get("titles")
        if isinstance(titles_field, dict):
            title_candidates = _flatten_to_list(titles_field.get("title"))
        else:
            title_candidates = _flatten_to_list(titles_field)
        title = strip_html(title_candidates[0]).strip() if title_candidates else ""

        creators = _extract_people(resource.get("creators"))
        contributors = _extract_people(resource.get("contributors"))
        creators_all: List[str] = []
        seen: Set[str] = set()
        for name in creators + contributors:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                creators_all.append(name)

        subjects = _clean_string_list(resource.get("subjects"))
        keywords = list(subjects)

        description = ""
        for field in ("description", "descriptions"):
            desc_field = resource.get(field)
            if desc_field:
                description = " ".join(_flatten_to_list(desc_field))
                break
        description = strip_html(description)

        year = None
        for candidate in _flatten_to_list(resource.get("date")) + _flatten_to_list(resource.get("publicationDate")):
            year = year_from_date(candidate)
            if year is not None:
                break

        types = _clean_string_list(resource.get("resourceType"))
        languages = _clean_string_list(resource.get("language"))
        formats = _clean_string_list(resource.get("format"))
        rights = _clean_string_list(resource.get("rights"))

        identifier_primary = resource.get("identifier")
        alternate_ids = []
        alt_field = resource.get("alternateIdentifiers")
        if isinstance(alt_field, dict):
            alternate_ids = _flatten_to_list(alt_field.get("alternateIdentifier"))
        else:
            alternate_ids = _flatten_to_list(alt_field)

        file_links = [link for link in _flatten_to_list(resource.get("file")) if link]
        relation_links = _clean_string_list(resource.get("relation"))

        identifiers = []
        if identifier_primary:
            identifiers.append(str(identifier_primary))
        identifiers.extend([i for i in alternate_ids if i])
        identifiers = [strip_html(i).strip() for i in identifiers if strip_html(i).strip()]

        relations = file_links + relation_links

        doi = extract_doi(identifier_primary, alternate_ids, relations, description)
        landing = pick_landing_url([identifier_primary], alternate_ids, file_links)

        extra: Dict[str, Any] = {}
        for key in ("resourceType", "alternateIdentifiers", "citationTitle", "publisher", "source", "file"):
            if resource.get(key):
                extra[key] = resource.get(key)

        out.append(Record(
            id=str(header_id),
            repo=repo_name,
            title=title,
            creators=creators_all,
            subjects=subjects,
            description=description,
            year=year,
            types=types,
            languages=languages,
            formats=formats,
            rights=rights,
            keywords=keywords,
            doi=doi,
            landing_url=landing,
            identifiers=identifiers,
            relations=relations,
            extra=extra,
        ))
    return out


def merge_record_lists(base: List[Record], extras: List[Record]) -> List[Record]:
    if not extras:
        return list(base)
    merged = list(base)
    existing_ids: Set[str] = {r.citation_id for r in base}
    for rec in extras:
        cid = rec.citation_id
        if cid in existing_ids and rec.doi:
            rec.extra.setdefault("original_doi", rec.doi)
            if rec.doi not in rec.identifiers:
                rec.identifiers.append(rec.doi)
            rec.doi = None
            cid = rec.citation_id
        if cid in existing_ids and rec.landing_url:
            rec.extra.setdefault("original_landing_url", rec.landing_url)
            rec.landing_url = f"{rec.landing_url}#repo={rec.repo}"
            cid = rec.citation_id
        if cid in existing_ids:
            base_id = rec.id or f"{rec.repo}:{len(merged)}"
            suffix = 1
            while rec.citation_id in existing_ids:
                rec.id = f"{base_id}#{suffix}"
                suffix += 1
        merged.append(rec)
        existing_ids.add(rec.citation_id)
    return merged



# ---------------------------
# 3) Build the graph (property graph, in-memory)
# ---------------------------

def _node_id(prefix: str, value: str) -> str:
    return f"{prefix}::{value}"


def build_graph(records: List[Record]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for r in records:
        rid = _node_id("dataset", r.citation_id)
        G.add_node(rid, label="Dataset", citation_id=r.citation_id, **asdict(r))

        # Authors
        for a in r.creators:
            al = a.lower()
            aid = _node_id("person", al)
            if not G.has_node(aid):
                G.add_node(aid, label="Person", name=a)
            G.add_edge(rid, aid, type="CREATED_BY")

        # Subjects / Keywords
        for s in r.subjects:
            sl = s.lower()
            sid = _node_id("subject", sl)
            if not G.has_node(sid):
                G.add_node(sid, label="Subject", name=s)
            G.add_edge(rid, sid, type="HAS_SUBJECT")

        # Relations (landing & DOI as nodes for explicit expansion)
        if r.doi:
            did = _node_id("doi", r.doi.lower())
            if not G.has_node(did):
                G.add_node(did, label="DOI", value=r.doi)
            G.add_edge(rid, did, type="HAS_DOI")

        if r.landing_url:
            lid = _node_id("url", r.landing_url)
            if not G.has_node(lid):
                G.add_node(lid, label="URL", value=r.landing_url)
            G.add_edge(rid, lid, type="HAS_URL")

        # Extra referenced publications (ubma_publications may contain HTML anchors)
        pubs = to_list(r.extra.get("ubma_publications"))
        for p in pubs:
            link = None
            m = re.search(r"href=['\"]([^'\"]+)['\"]", p or "", flags=re.I)
            if m:
                link = m.group(1)
            txt = strip_html(p or "")
            pid_raw = (link or txt)[:200].lower()
            pid = _node_id("pub", pid_raw)
            if not G.has_node(pid):
                G.add_node(pid, label="Publication", title=txt, url=link)
            G.add_edge(rid, pid, type="MENTIONS_PUBLICATION")

        # Languages / Formats (optional, can be useful for filtering)
        for lang in r.languages:
            if not lang:
                continue
            lgid = _node_id("lang", str(lang).lower())
            if not G.has_node(lgid):
                G.add_node(lgid, label="Language", code=lang)
            G.add_edge(rid, lgid, type="HAS_LANGUAGE")

        for fmt in r.formats:
            if not fmt:
                continue
            fid = _node_id("fmt", str(fmt).lower())
            if not G.has_node(fid):
                G.add_node(fid, label="Format", name=fmt)
            G.add_edge(rid, fid, type="HAS_FORMAT")

    return G


# ---------------------------
# 4) Text indices: TF-IDF + Dense (SentenceTransformer + FAISS/Numpy) with caching
# ---------------------------

class DualRetriever:
    def __init__(
        self,
        records: List[Record],
        emb_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir: Optional[str | os.PathLike[str]] = None,
    ):
        self.records = records
        self.ids = [r.citation_id for r in records]
        self.corpus = [r.index_text() for r in records]

        # Sparse TF-IDF
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.9)
        self.tfidf = self.vectorizer.fit_transform(self.corpus)

        # Dense embeddings with caching
        self.embedder = SentenceTransformer(emb_model)
        self.used_embedding_cache = False
        self.emb = self._load_or_build_embeddings(emb_model, cache_dir)

        # FAISS or NumPy index
        if FAISS_OK:
            d = self.emb.shape[1]
            self.faiss_index = faiss.IndexFlatIP(d)  # cosine if normalized
            self.faiss_index.add(self.emb.astype(np.float32))
        else:
            self.faiss_index = None  # will use NumPy similarity

        self.id2pos = {cid: i for i, cid in enumerate(self.ids)}
        self.pos2id = {i: cid for i, cid in enumerate(self.ids)}

    def _signature(self) -> str:
        h = hashlib.sha1()
        for cid, text in zip(self.ids, self.corpus):
            h.update(cid.encode("utf-8"))
            h.update(b"\x00")
            h.update(text[:2000].encode("utf-8"))  # partial content for speed
            h.update(b"\x01")
        return h.hexdigest()

    def _cache_paths(self, emb_model: str, cache_dir: Optional[str | os.PathLike[str]]) -> Tuple[Optional[Path], Optional[Path]]:
        if cache_dir is None:
            return None, None
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        sig = self._signature()
        safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", emb_model)
        npy_path = cache_root / f"emb_{safe_model}_{sig}.npy"
        meta_path = cache_root / f"emb_{safe_model}_{sig}.json"
        return npy_path, meta_path

    def _load_or_build_embeddings(self, emb_model: str, cache_dir: Optional[str | os.PathLike[str]]):
        npy_path, meta_path = self._cache_paths(emb_model, cache_dir)
        if npy_path and meta_path and npy_path.exists() and meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("num") == len(self.corpus):
                    self.used_embedding_cache = True
                    return np.load(npy_path)
            except Exception:
                pass
        # build fresh
        self.used_embedding_cache = False
        emb = self.embedder.encode(
            self.corpus,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        if npy_path and meta_path:
            try:
                np.save(npy_path, emb)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump({"num": len(self.corpus)}, f)
            except Exception:
                pass
        return emb

    def dense_search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        qv = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        if self.faiss_index is not None:
            D, I = self.faiss_index.search(qv.reshape(1,-1).astype(np.float32), top_k)
            return [(self.pos2id[int(i)], float(D[0][j])) for j, i in enumerate(I[0])]
        # fallback: numpy
        sims = (self.emb @ qv)
        idx = np.argsort(-sims)[:top_k]
        return [(self.pos2id[int(i)], float(sims[i])) for i in idx]

    def sparse_search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.tfidf)[0]
        idx = np.argsort(-sims)[:top_k]
        return [(self.pos2id[int(i)], float(sims[i])) for i in idx]

    @staticmethod
    def rrf(rank_lists: List[List[str]], k: int = 60) -> List[str]:
        # Reciprocal Rank Fusion over ID lists
        scores = defaultdict(float)
        for lst in rank_lists:
            for rank, cid in enumerate(lst, start=1):
                scores[cid] += 1.0 / (k + rank)
        return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]

    def hybrid_search(self, query: str, top_k_sparse=50, top_k_dense=50, final_k=50) -> List[str]:
        s = self.sparse_search(query, top_k_sparse)
        d = self.dense_search(query, top_k_dense)
        s_ids = [cid for cid, _ in s]
        d_ids = [cid for cid, _ in d]
        fused = self.rrf([s_ids, d_ids])
        return fused[:final_k]


# ---------------------------
# 5) Graph expansion + packing context
# ---------------------------

def _records_to_jsonable(records: List[Record]) -> List[Dict[str, Any]]:
    return [asdict(r) for r in records]



def _normalize_category_name(name: str) -> str:
    n = name.lower()
    if n.startswith('publication'):
        return 'publication'
    if n.startswith('dataset'):
        return 'dataset'
    return n


def _record_category(rec: Record) -> str:
    types = [t.lower() for t in rec.types]
    if rec.repo.lower() == 'madata' or 'dataset' in types:
        return 'dataset'
    if rec.repo.lower() == 'madoc' or any('publication' in t for t in types):
        return 'publication'
    return 'other'


def _quantity_requests(q: str) -> Dict[str, int]:
    matches = re.findall(r"(\d{1,3})\s+(publications?|datasets?)", q, flags=re.I)
    req: Dict[str, int] = {}
    for num, label in matches:
        key = _normalize_category_name(label)
        req[key] = max(req.get(key, 0), int(num))
    return req


def _all_requests(q: str) -> Set[str]:
    matches = re.findall(r"\ball\s+(publications?|datasets?)\b", q, flags=re.I)
    return {_normalize_category_name(label) for label in matches}

def _records_from_jsonable(objs: List[Dict[str, Any]]) -> List[Record]:
    out: List[Record] = []
    for o in objs:
        out.append(Record(**o))
    return out


def _results_signature(results: List[Dict[str, Any]]) -> str:
    h = hashlib.sha1()
    h.update(str(len(results)).encode("utf-8"))
    for i, rec in enumerate(results[:200]):
        header_id = safe_get(rec, "header", "identifier") or ""
        h.update(str(i).encode("utf-8"))
        h.update(str(header_id).encode("utf-8"))
    return h.hexdigest()


QUERY_STOPWORDS = {
    "provide", "provided", "providing", "show", "list", "give", "find", "tell",
    "answer", "question", "publication", "publications", "dataset", "datasets",
    "information", "summary", "detail", "details", "about", "with", "from", "for",
    "and", "the", "please", "top", "best", "latest", "new", "by", "of", "on", "all"
}

class GraphRAG:
    def __init__(self, records: List[Record], graph: nx.MultiDiGraph, retriever: DualRetriever):
        self.records = {r.citation_id: r for r in records}
        self.G = graph
        self.R = retriever

    @classmethod
    def from_records(
        cls,
        records: List[Record],
        emb_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir: Optional[str | os.PathLike[str]] = None,
    ):
        G = build_graph(records)
        R = DualRetriever(records, emb_model=emb_model, cache_dir=cache_dir)
        return cls(records, G, R)

    @classmethod
    def from_oai_dc(
        cls,
        results: List[Dict[str, Any]],
        repo_name: str = "madata",
        emb_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir: Optional[str | os.PathLike[str]] = None,
    ):
        recs = normalize_oai_dc(results, repo_name=repo_name)
        G = build_graph(recs)
        R = DualRetriever(recs, emb_model=emb_model, cache_dir=cache_dir)
        return cls(recs, G, R)

    def _dataset_node(self, cid: str) -> Optional[str]:
        nid = f"dataset::{cid}"
        return nid if self.G.has_node(nid) else None

    def _dataset_ids_by_person_substring(self, needle: str) -> Set[str]:
        out: Set[str] = set()
        if not needle:
            return out
        # token-based match, order-insensitive
        tokens = [t for t in re.findall(r"[\w']+", needle.lower()) if len(t) >= 3]
        if not tokens:
            return out
        for n, data in self.G.nodes(data=True):
            if data.get("label") == "Person":
                name = str(data.get("name") or "").lower()
                # require at least one token to match
                if any(t in name for t in tokens):
                    for ds in self.G.predecessors(n):
                        if self.G.nodes[ds].get("label") == "Dataset":
                            cid = self.G.nodes[ds].get("citation_id")
                            if cid:
                                out.add(cid)
        return out

    def _dataset_ids_by_subject_substring(self, needle: str) -> Set[str]:
        out: Set[str] = set()
        if not needle:
            return out
        nlow = needle.lower()
        for n, data in self.G.nodes(data=True):
            if data.get("label") == "Subject":
                name = str(data.get("name") or "").lower()
                if nlow in name:
                    for ds in self.G.predecessors(n):
                        if self.G.nodes[ds].get("label") == "Dataset":
                            cid = self.G.nodes[ds].get("citation_id")
                            if cid:
                                out.add(cid)
        return out

    def _boost_from_query_entities(self, q: str) -> Set[str]:
        boosted: Set[str] = set()
        name_matches = re.findall(r"([A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+)+)", q)
        for name in name_matches:
            boosted |= self._dataset_ids_by_person_substring(name)
        tokens = [t for t in re.findall(r"[\w'-]+", q) if len(t) >= 3]
        for t in tokens:
            tl = t.lower()
            if tl in QUERY_STOPWORDS:
                continue
            boosted |= self._dataset_ids_by_person_substring(t)
        return boosted

    @staticmethod
    def _years_from_query(q: str) -> List[int]:
        yrs = [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", q) if isinstance(y, str)]
        # the regex above captures only the century group; fix to capture full year
        yrs2 = [int(y) for y in re.findall(r"\b(\d{4})\b", q) if y.startswith(("19", "20"))]
        return yrs2 or yrs

    def expand_neighbors(self, seed_ids: List[str], hops: int = 1, max_nodes: int = 200) -> Set[str]:
        """Return set of dataset citation_ids after expanding neighbors around seeds."""
        visited_nodes: Set[str] = set()
        frontier: Set[str] = set()
        for cid in seed_ids:
            nid = self._dataset_node(cid)
            if nid:
                frontier.add(nid)
                visited_nodes.add(nid)

        for _ in range(hops):
            next_frontier: Set[str] = set()
            for n in list(frontier):
                for _, nbr, _edata in self.G.out_edges(n, data=True):
                    next_frontier.add(nbr)
                for nbr, _, _edata in self.G.in_edges(n, data=True):
                    next_frontier.add(nbr)
            visited_nodes |= next_frontier
            frontier = next_frontier
            if len(visited_nodes) > max_nodes:
                break

        # Keep only dataset nodes → return citation_ids
        out: Set[str] = set()
        for n in visited_nodes:
            if self.G.nodes[n].get("label") == "Dataset":
                cid = self.G.nodes[n].get("citation_id")
                if cid:
                    out.add(cid)
            # Also pull datasets that share authors/subjects with seeds
            if self.G.nodes[n].get("label") in ("Person", "Subject"):
                for ds in self.G.predecessors(n):
                    if self.G.nodes[ds].get("label") == "Dataset":
                        cid = self.G.nodes[ds].get("citation_id")
                        if cid:
                            out.add(cid)
        return out

    def fetch_records(self, ids: List[str]) -> List[Record]:
        return [self.records[i] for i in ids if i in self.records]

    def pack_context(self, recs: List[Record], max_chars: int = 9000) -> str:
        """Assemble a structured context with strict citations."""
        blocks = []
        for r in recs:
            b = [
                f"- id: {r.citation_id} | repo: {r.repo} | type: {', '.join(r.types) or 'Dataset'}",
                f"  title: {r.title}",
                f"  authors: {', '.join(r.creators)}{f' ({r.year})' if r.year else ''}",
                f"  subjects: {', '.join(r.subjects)}",
                f"  description: {strip_html(r.description)}",
                f"  link: {r.doi or r.landing_url or ''}",
            ]
            blocks.append("\n".join(b))
        text = "\n\n".join(blocks)
        # Trim to model-friendly size
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…"
        return text

    def query(
        self,
        q: str,
        seed_k: int = 30,
        expand_hops: int = 1,
        final_k: int = 10,
        author_filters: Optional[List[str]] = None,
        subject_filters: Optional[List[str]] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> Tuple[str, List[Record]]:
        # 0) optional structured filters and query-entity boosts
        author_filters = author_filters or []
        subject_filters = subject_filters or []

        # infer years from query if not provided explicitly
        inferred_years = self._years_from_query(q)
        inferred_range: Optional[Tuple[int, int]] = None
        if year_from or year_to:
            pass
        elif len(inferred_years) == 1:
            year_from = year_to = inferred_years[0]
            inferred_range = (year_from, year_to)
        elif len(inferred_years) >= 2:
            y1, y2 = min(inferred_years), max(inferred_years)
            year_from, year_to = y1, y2
            inferred_range = (year_from, year_to)

        def _year_ok(r: Record) -> bool:
            if year_from is None and year_to is None:
                return True
            if r.year is None:
                return False
            if year_from is not None and r.year < year_from:
                return False
            if year_to is not None and r.year > year_to:
                return False
            return True

        filter_ids: Set[str] = set()
        for a in author_filters:
            filter_ids |= self._dataset_ids_by_person_substring(a)
        for s in subject_filters:
            filter_ids |= self._dataset_ids_by_subject_substring(s)

        boosted_from_query = self._boost_from_query_entities(q)
        quantity_requests = _quantity_requests(q)
        all_requests = _all_requests(q)
        required_total = sum(quantity_requests.values())
        target_total = max(final_k, required_total) if required_total else final_k

        # 1) hybrid retrieve seeds
        seed_ids = self.R.hybrid_search(q, final_k=seed_k)
        if filter_ids:
            seed_ids = list(dict.fromkeys(list(filter_ids) + seed_ids))  # put filters first, keep order/uniques

        # 2) graph expansion around seeds
        cand_ids = list(self.expand_neighbors(seed_ids, hops=expand_hops))

        # 3) local re-rank of candidates against the query using cosine over embeddings
        if not cand_ids:
            cand_ids = seed_ids
        cand_vecs = []
        cand_order = []
        for cid in cand_ids:
            pos = self.R.id2pos.get(cid)
            if pos is None:
                continue
            r = self.records.get(cid)
            if r is None or not _year_ok(r):
                continue
            cand_vecs.append(self.R.emb[pos])
            cand_order.append(cid)
        if not cand_vecs:
            # fallback to seeds (apply year filter softly if inferred)
            base = seed_ids
            tmp_order = []
            tmp_vecs = []
            for cid in base:
                pos = self.R.id2pos.get(cid)
                if pos is None:
                    continue
                r = self.records.get(cid)
                if r is None:
                    continue
                if (year_from or year_to) and not _year_ok(r):
                    continue
                tmp_vecs.append(self.R.emb[pos])
                tmp_order.append(cid)
            cand_order = tmp_order
            cand_vecs = tmp_vecs

        # Enforce hard filters when provided
        if (author_filters or subject_filters) and cand_order and filter_ids:
            allowed = filter_ids
            filtered_order = [cid for cid in cand_order if cid in allowed]
            filtered_vecs = [vec for cid, vec in zip(cand_order, cand_vecs) if cid in allowed]
            if filtered_order:
                cand_order = filtered_order
                cand_vecs = filtered_vecs
            # else: if no allowed after expansion, keep current list and rely on boosts

        # If no explicit author filters, but query mentions a person and we have matches,
        # restrict to those matches to avoid drift.
        if not author_filters and boosted_from_query and cand_order:
            matched_order = [cid for cid in cand_order if cid in boosted_from_query]
            if matched_order:
                cand_vecs = [vec for cid, vec in zip(cand_order, cand_vecs) if cid in boosted_from_query]
                cand_order = matched_order

        ranked_pairs: List[Tuple[str, float]] = []
        if cand_vecs:
            qv = self.R.embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
            sims = np.array([float(np.dot(v, qv)) for v in cand_vecs])
            boosts = np.array([
                (0.2 if cid in boosted_from_query else 0.0) + (0.4 if cid in filter_ids else 0.0)
                for cid in cand_order
            ])
            final_scores = sims + boosts
            order = np.argsort(-final_scores)
            ranked_pairs = [(cand_order[i], float(final_scores[i])) for i in order]
        else:
            ranked_pairs = [(cid, 0.0) for cid in cand_order]

        if all_requests:
            existing_ids = {cid for cid, _ in ranked_pairs}
            for cid in boosted_from_query:
                if cid in existing_ids:
                    continue
                rec = self.records.get(cid)
                if rec and _record_category(rec) in all_requests and _record_matches_query(rec, q):
                    ranked_pairs.append((cid, 0.0))
                    existing_ids.add(cid)

            if ranked_pairs:
                available_counts: Dict[str, int] = {}
                for cid, _ in ranked_pairs:
                    rec = self.records.get(cid)
                    if rec is None or not _record_matches_query(rec, q):
                        continue
                    cat = _record_category(rec)
                    if cat in all_requests:
                        available_counts[cat] = available_counts.get(cat, 0) + 1
                for cat in all_requests:
                    total = available_counts.get(cat, 0)
                    if total:
                        quantity_requests[cat] = max(quantity_requests.get(cat, 0), total)
                required_total = sum(quantity_requests.values())
                if required_total:
                    target_total = max(target_total, required_total)

        if not ranked_pairs:
            return "", []

        selected: List[Tuple[str, float, Record]] = []
        used: Set[str] = set()

        if quantity_requests:
            remaining = dict(quantity_requests)
            for cid, score in ranked_pairs:
                if len(selected) >= target_total:
                    break
                if cid in used:
                    continue
                rec = self.records.get(cid)
                if rec is None or not _record_matches_query(rec, q):
                    continue
                category = _record_category(rec)
                if remaining.get(category, 0) > 0:
                    selected.append((cid, score, rec))
                    used.add(cid)
                    remaining[category] = max(remaining.get(category, 0) - 1, 0)
            for cid, score in ranked_pairs:
                if len(selected) >= target_total:
                    break
                if cid in used:
                    continue
                rec = self.records.get(cid)
                if rec is None or not _record_matches_query(rec, q):
                    continue
                selected.append((cid, score, rec))
                used.add(cid)
        else:
            for cid, score in ranked_pairs:
                if len(selected) >= target_total:
                    break
                if cid in used:
                    continue
                rec = self.records.get(cid)
                if rec is None or not _record_matches_query(rec, q):
                    continue
                selected.append((cid, score, rec))
                used.add(cid)

        top_recs = [rec for _, _, rec in selected]
        context = self.pack_context(top_recs)
        return context, top_recs


# ---------------------------
# 6) (Optional) Local generation with Ollama
# ---------------------------

OLLAMA_SYSTEM = """You are a librarian assistant.
Follow these rules strictly:
- Answer ONLY using the provided CONTEXT.
- Only include datasets that DIRECTLY answer the user's request; omit unrelated items.
- If the user question is broad, return a concise list of the most relevant datasets.
- For EVERY factual item (title, author, year, claim), add a citation in square brackets using its id (DOI or URL). Example: [10.7801/123].
- If the answer cannot be supported by the context, say you don’t know.
- Be concise. Prefer bullet points when listing datasets.
"""


def _record_matches_query(r: "Record", q: str) -> bool:
    raw_tokens = [t for t in re.findall(r"[\w'-]+", q) if len(t) >= 3]
    name_matches = re.findall(r"\b([A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+)*)\b", q)

    creators_lower = [normalize_name(c).lower() for c in r.creators]
    if name_matches:
        name_ok = False
        for name in name_matches:
            parts = [p for p in re.split(r"\s+", name.strip()) if len(p) >= 3]
            for part in parts:
                pl = part.lower()
                if any(pl in creator for creator in creators_lower):
                    name_ok = True
                    break
            if name_ok:
                break
        if not name_ok:
            return False

    content_tokens = [t.lower() for t in raw_tokens if t.lower() not in QUERY_STOPWORDS]
    if not content_tokens:
        return bool(name_matches)

    hay = " \n ".join([
        r.title.lower(),
        " ".join(creators_lower),
        " ".join([s.lower() for s in r.subjects]),
        strip_html(r.description).lower(),
    ])
    return any(tok in hay for tok in content_tokens)


def answer_with_ollama(context: str, question: str, model: str = "gpt-oss:20b", *, answer_style: str = "list", max_ctx_chars: int = 9000) -> str:
    """
    Requires local Ollama running. If you prefer another local LLM server,
    replace this function with your client call.
    """
    import subprocess, os

    # Trim context if requested
    if len(context) > max_ctx_chars:
        context = context[:max_ctx_chars] + "\n…"

    instructions = (
        f"ANSWER STYLE: {answer_style}\n"
        "- list: list top datasets with title, authors, year, and link with citations.\n"
        "- summary: short paragraph summarizing the key datasets with citations.\n"
        "- qa: directly answer the question with citations.\n"
    )

    prompt = f"{OLLAMA_SYSTEM}\n\n{instructions}\nUSER QUERY:\n{question}\n\nCONTEXT:\n{context}"

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Ensure standard Ollama host if not set
    env.setdefault("OLLAMA_HOST", "127.0.0.1:11434")

    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        return f"(Local LLM error: {proc.stderr.decode('utf-8')})"
    return proc.stdout.decode("utf-8").strip()


# ---------------------------
# 7) CLI / Example usage
# ---------------------------

def _print_top(recs: List[Record], k: int = 5) -> None:
    for r in recs[:k]:
        link = r.doi or r.landing_url or r.id
        authors = ", ".join(r.creators)
        print(f"• {r.title} | Authors: {authors} | Link: {link}")


def _print_grouped_by_category(recs: List[Record]) -> None:
    groups: Dict[str, List[Record]] = {"publication": [], "dataset": [], "other": []}
    for r in recs:
        groups.setdefault(_record_category(r), []).append(r)

    labels = {
        "publication": "Publications",
        "dataset": "Datasets",
        "other": "Other Resources",
    }

    for key in ("publication", "dataset", "other"):
        bucket = groups.get(key) or []
        if not bucket:
            continue
        print(f"{labels[key]} ({len(bucket)}):")
        for r in bucket:
            link = r.doi or r.landing_url or r.id
            authors = ", ".join(r.creators)
            print(f"  • {r.title} | Authors: {authors or 'n/a'} | Link: {link}")


def _record_to_minimal_dict(r: Record) -> Dict[str, Any]:
    return {
        "id": r.citation_id,
        "repo": r.repo,
        "title": r.title,
        "authors": r.creators,
        "year": r.year,
        "subjects": r.subjects,
        "description": strip_html(r.description),
        "types": r.types,
        "languages": r.languages,
        "formats": r.formats,
        "rights": r.rights,
        "keywords": r.keywords,
        "doi": r.doi,
        "landing_url": r.landing_url,
        "identifiers": r.identifiers,
        "relations": r.relations,
    }


def _format_person_for_citation(name: str) -> str:
    # Accept forms like "Last, First Middle" or "First Middle Last"
    n = name.strip()
    if "," in n:
        last, rest = [p.strip() for p in n.split(",", 1)]
    else:
        parts = n.split()
        if len(parts) == 1:
            return n
        last = parts[-1]
        rest = " ".join(parts[:-1])
    initials = " ".join([p[0] + "." for p in rest.split() if p and p[0].isalpha()])
    return f"{last}, {initials}".strip().rstrip(",")


def _authors_citation(creators: List[str], max_authors: int = 6) -> str:
    if not creators:
        return ""
    formatted = [_format_person_for_citation(a) for a in creators]
    if len(formatted) <= max_authors:
        if len(formatted) == 2:
            return f"{formatted[0]} & {formatted[1]}"
        if len(formatted) > 2:
            return ", ".join(formatted[:-1]) + f" & {formatted[-1]}"
        return formatted[0]
    # many authors: list first N then et al.
    head = ", ".join(formatted[:max_authors])
    return f"{head} et al."


def _print_citations(recs: List[Record]) -> None:
    for r in recs:
        authors = _authors_citation(r.creators)
        year = f" ({r.year})" if r.year else ""
        title = r.title.rstrip(".")
        link = r.doi or r.landing_url or r.id
        # Simple GS-like style: Authors (Year). Title. DOI
        if authors:
            print(f"{authors}{year}. {title}. {link}")
        else:
            print(f"{title}{year}. {link}")


def _print_brief(recs: List[Record]) -> None:
    for r in recs:
        print(f"• {r.title} | Link: {r.doi or r.landing_url or r.id}")


def _print_full(recs: List[Record], *, desc_chars: int = 0) -> None:
    for r in recs:
        print(f"- id: {r.citation_id} | repo: {r.repo}")
        print(f"  title: {r.title}")
        if r.creators:
            print(f"  authors: {', '.join(r.creators)}{f' ({r.year})' if r.year else ''}")
        elif r.year is not None:
            print(f"  year: {r.year}")
        if r.subjects:
            print(f"  subjects: {', '.join(r.subjects)}")
        if r.types:
            print(f"  types: {', '.join(r.types)}")
        if r.languages:
            print(f"  languages: {', '.join(map(str, r.languages))}")
        if r.formats:
            print(f"  formats: {', '.join(map(str, r.formats))}")
        if r.rights:
            print(f"  rights: {', '.join(r.rights)}")
        if r.keywords:
            print(f"  keywords: {', '.join(r.keywords)}")
        if r.description:
            d = strip_html(r.description)
            if desc_chars and len(d) > desc_chars:
                d = d[:desc_chars] + "…"
            print(f"  description: {d}")
        print(f"  link: {r.doi or r.landing_url or ''}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GraphRAG over MADATA OAI-DC")
    sub = parser.add_subparsers(dest="cmd", required=False)

    # harvest command
    p_h = sub.add_parser("harvest", help="Harvest from OAI-PMH and save JSON")
    p_h.add_argument("--out", required=True, help="Output JSON path")
    p_h.add_argument("--base-url", default="https://madata.bib.uni-mannheim.de/cgi/oai2")
    p_h.add_argument("--prefix", default="oai_dc")
    p_h.add_argument("--set", dest="set_spec", default=None)
    p_h.add_argument("--max", dest="max_records", type=int, default=None)
    p_h.add_argument("--sleep", dest="sleep_secs", type=float, default=0.0)
    p_h.add_argument("--force", action="store_true", help="Force re-harvest even if output exists from today")
    p_h.add_argument("--quiet", action="store_true", help="Reduce status output; still prints results/answers")

    # query command
    p_q = sub.add_parser("query", help="Load JSON, build indices, and query")
    p_q.add_argument("--results", required=True, help="Path to harvested JSON")
    p_q.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p_q.add_argument("--cache", default=".graphrag_cache")
    p_q.add_argument("--seed-k", type=int, default=30)
    p_q.add_argument("--expand-hops", type=int, default=1)
    p_q.add_argument("--final-k", type=int, default=8)
    p_q.add_argument("--question", required=True)
    p_q.add_argument("--author", nargs="+", help="Filter by author name (substring match)")
    p_q.add_argument("--subject", nargs="+", help="Filter by subject name (substring match)")
    p_q.add_argument("--year", type=int, help="Filter to a single publication year")
    p_q.add_argument("--year-from", dest="year_from", type=int, help="Filter to records with year >= this")
    p_q.add_argument("--year-to", dest="year_to", type=int, help="Filter to records with year <= this")
    p_q.add_argument("--ollama", action="store_true", help="Use Ollama to generate an answer from retrieved context")
    p_q.add_argument("--ollama-model", default="gpt-oss:20b", help="Ollama model to use")
    p_q.add_argument("--full", action="store_true", help="Print full bibliographic records")
    p_q.add_argument("--json", dest="as_json", action="store_true", help="Output results as JSON records")
    p_q.add_argument("--quiet", action="store_true", help="Reduce status output; still prints results/answers")
    p_q.add_argument("--brief", action="store_true", help="Print only title and link")
    p_q.add_argument("--desc-chars", type=int, default=0, help="When --full, truncate description to this many characters")
    p_q.add_argument("--json-fields", type=str, default="", help="Comma-separated subset of fields for --json output")
    p_q.add_argument("--cite", action="store_true", help="Print Google Scholar-like citations with DOI/URL")
    p_q.add_argument("--madoc-results", type=str, default="", help="Path to MADOC OpenAIRE harvested JSON to enrich the graph")

    # grep command
    p_g = sub.add_parser("grep", help="Search harvested JSON for a substring and show matching fields")
    p_g.add_argument("--results", required=True)
    p_g.add_argument("--pattern", required=True)
    p_g.add_argument("--limit", type=int, default=10)

    # list-authors command
    p_la = sub.add_parser("list-authors", help="List top normalized authors from results")
    p_la.add_argument("--results", required=True)
    p_la.add_argument("--top", type=int, default=50)

    # madoc command
    p_madoc = sub.add_parser("madoc", help="Harvest MADOC OpenAIRE records and save JSON")
    p_madoc.add_argument("--out", required=True, help="Output JSON path")
    p_madoc.add_argument("--base-url", default="https://madoc.bib.uni-mannheim.de/cgi/oai2")
    p_madoc.add_argument("--prefix", default="oai_openaire")
    p_madoc.add_argument("--set", dest="set_spec", default=None)
    p_madoc.add_argument("--max", dest="max_records", type=int, default=None)
    p_madoc.add_argument("--sleep", dest="sleep_secs", type=float, default=0.0)
    p_madoc.add_argument("--force", action="store_true", help="Force re-harvest even if output exists from today")
    p_madoc.add_argument("--quiet", action="store_true", help="Reduce status output; still prints results/answers")

    # default behavior if no subcommand: show a quick interactive using live harvest limit
    parser.add_argument("--quick", action="store_true", help="Quick mode: live harvest a handful then query")
    parser.add_argument("--quiet", action="store_true", help="Reduce status output; still prints results/answers")
    parser.add_argument("--question", help="Question for quick mode")
    parser.add_argument("--max", type=int, default=50, help="Max records for quick mode")

    args = parser.parse_args()

    if args.cmd == "harvest":
        # Skip if already harvested today unless --force
        try:
            outp = Path(args.out)
            if outp.exists() and not args.force:
                import datetime
                mdate = datetime.datetime.fromtimestamp(outp.stat().st_mtime).date()
                if mdate == datetime.date.today():
                    if not args.quiet:
                        print(f"Output exists and was created today: {args.out}. Skipping harvest. Use --force to override.")
                    return
        except Exception:
            pass

        results = harvest_oai_dc(
            base_url=args.base_url,
            metadata_prefix=args.prefix,
            set_spec=args.set_spec,
            max_records=args.max_records,
            sleep_secs=args.sleep_secs,
        )
        save_results_json(results, args.out)
        if not args.quiet:
            print(f"Saved {len(results)} records to {args.out}")
        return

    if args.cmd == "query":
        results = load_results_json(args.results)
        if not args.quiet:
            print("Preparing records and indices…")
        cache_root = Path(args.cache)
        cache_root.mkdir(parents=True, exist_ok=True)
        sig = _results_signature(results)
        norm_path = cache_root / f"norm_{sig}.json"
        use_cached_norm = False
        try:
            if norm_path.exists():
                import datetime
                mdate = datetime.datetime.fromtimestamp(norm_path.stat().st_mtime).date()
                if mdate == datetime.date.today():
                    use_cached_norm = True
        except Exception:
            pass

        recs: Optional[List[Record]] = None
        if use_cached_norm:
            try:
                with open(norm_path, "r", encoding="utf-8") as f:
                    rec_objs = json.load(f)
                recs = _records_from_jsonable(rec_objs)
                if not args.quiet:
                    print(f"Using cached normalized records: {norm_path}")
            except Exception as e:
                recs = None
                if not args.quiet:
                    print(f"[warn] Failed to load normalized cache: {e}")

        if recs is None:
            recs = normalize_oai_dc(results, repo_name="madata")
            try:
                with open(norm_path, "w", encoding="utf-8") as f:
                    json.dump(_records_to_jsonable(recs), f, ensure_ascii=False)
                if not args.quiet:
                    print(f"Saved normalized records cache: {norm_path}")
            except Exception as e:
                if not args.quiet:
                    print(f"[warn] Failed to write normalized cache: {e}")

        madoc_recs: List[Record] = []
        if args.madoc_results:
            try:
                madoc_res = load_results_json(args.madoc_results)
                madoc_recs = normalize_openaire(madoc_res, repo_name="madoc")
                if not args.quiet:
                    print(f"Loaded {len(madoc_recs)} MADOC publications from {args.madoc_results}.")
            except Exception as e:
                madoc_recs = []
                if not args.quiet:
                    print(f"[warn] Failed to load MADOC publications: {e}")

        if madoc_recs:
            before = len(recs)
            recs = merge_record_lists(recs, madoc_recs)
            added = len(recs) - before
            if not args.quiet:
                print(f"Merged {added} MADOC publications into record set.")

        graphrag = GraphRAG.from_records(recs, emb_model=args.model, cache_dir=args.cache)

        if not args.quiet:
            if getattr(graphrag.R, "used_embedding_cache", False):
                print("Using cached embeddings.")
            else:
                print("Built embeddings (cached for future runs).")

        print(f"Records indexed: {len(graphrag.records)} | Graph nodes: {graphrag.G.number_of_nodes()} | edges: {graphrag.G.number_of_edges()}")
        ctx, recs = graphrag.query(
            args.question,
            seed_k=args.seed_k,
            expand_hops=args.expand_hops,
            final_k=args.final_k,
            author_filters=args.author,
            subject_filters=args.subject,
            year_from=(args.year if args.year is not None else args.year_from),
            year_to=(args.year if args.year is not None else args.year_to),
        )
        quantity_requests = _quantity_requests(args.question or "")

        if args.as_json:
            items = [_record_to_minimal_dict(r) for r in recs]
            if args.json_fields:
                wanted = [f.strip() for f in args.json_fields.split(",") if f.strip()]
                if wanted:
                    items = [{k: v for k, v in item.items() if k in wanted} for item in items]
            print(json.dumps(items, ensure_ascii=False, indent=2))
        elif args.cite:
            _print_citations(recs)
        elif args.full:
            _print_full(recs, desc_chars=args.desc_chars)
        elif args.brief:
            _print_brief(recs)
        else:
            if quantity_requests:
                _print_grouped_by_category(recs)
            else:
                _print_top(recs, k=len(recs))
        if args.ollama:
            if not args.quiet:
                print("\nGenerating answer with Ollama…")
            try:
                # Filter context to records matching the question terms to avoid unrelated items
                filtered = [r for r in recs if _record_matches_query(r, args.question)]
                if filtered:
                    ctx = graphrag.pack_context(filtered)
                ans = answer_with_ollama(ctx, args.question, model=args.ollama_model, answer_style="list", max_ctx_chars=9000)
            except Exception as e:
                ans = f"(Error calling Ollama: {e})"
            print("\nANSWER:\n" + ans)
        return

    if args.cmd == "grep":
        import re as _re
        results = load_results_json(args.results)
        pat = _re.compile(args.pattern, _re.I)
        shown = 0
        for rec in results:
            dc = (_ensure_dict(rec.get("metadata")).get("dc")) or {}
            flat = json.dumps(dc, ensure_ascii=False)
            if pat.search(flat):
                fields = [k for k, v in dc.items() if pat.search(json.dumps(v, ensure_ascii=False))]
                print("FIELDS:", fields)
                if "title" in dc:
                    print("TITLE:", strip_html(str(dc.get("title"))))
                else:
                    print("TITLE:", "<no title>")
                print("CREATOR:", dc.get("creator"))
                print("CONTRIBUTOR:", dc.get("contributor"))
                print("UBMA_AUTHOR(S):", dc.get("ubma_author"), dc.get("ubma_authors"))
                print("---")
                shown += 1
                if shown >= args.limit:
                    break
        print(f"matches shown: {shown}")
        return

    if args.cmd == "list-authors":
        results = load_results_json(args.results)
        recs = normalize_oai_dc(results, repo_name="madata")
        from collections import Counter
        c = Counter()
        for r in recs:
            for a in r.creators:
                c[a] += 1
        for a, n in c.most_common(args.top):
            print(f"{n}\t{a}")
        return

    if args.cmd == "madoc":
        # Skip if already harvested today unless --force
        try:
            outp = Path(args.out)
            if outp.exists() and not args.force:
                import datetime
                mdate = datetime.datetime.fromtimestamp(outp.stat().st_mtime).date()
                if mdate == datetime.date.today():
                    if not args.quiet:
                        print(f"Output exists and was created today: {args.out}. Skipping harvest. Use --force to override.")
                    return
        except Exception:
            pass

        results = harvest_oai_dc(
            base_url=args.base_url,
            metadata_prefix=args.prefix,
            set_spec=args.set_spec,
            max_records=args.max_records,
            sleep_secs=args.sleep_secs,
        )
        save_results_json(results, args.out)
        if not args.quiet:
            print(f"Saved {len(results)} records to {args.out}")
        return

    # Quick mode
    if args.quick:
        if not args.question:
            raise SystemExit("--question is required for --quick mode")
        results = harvest_oai_dc(max_records=args.max)
        print("Normalizing and indexing records…")
        graphrag = GraphRAG.from_oai_dc(results, repo_name="madata", cache_dir=".graphrag_cache")
        print(f"Records indexed: {len(graphrag.records)} | Graph nodes: {graphrag.G.number_of_nodes()} | edges: {graphrag.G.number_of_edges()}")
        ctx, recs = graphrag.query(args.question, seed_k=30, expand_hops=1, final_k=8)
        _print_top(recs)
        return

    # If nothing chosen, print help
    parser.print_help()


if __name__ == "__main__":
    main()
