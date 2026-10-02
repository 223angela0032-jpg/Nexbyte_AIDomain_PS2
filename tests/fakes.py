"""
Stand-ins for packages that are not installable offline, so the project's OWN code can be
executed end to end:  streamlit, faiss, sentence_transformers, fitz (PyMuPDF), google.genai.
They mimic just the API surface the project uses.
"""
import hashlib
import re
import sys
import types
from contextlib import contextmanager

import numpy as np


# ------------------------------------------------------------------ streamlit
class Rerun(BaseException):
    pass


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeStreamlit(types.ModuleType):
    """Scripted Streamlit. Widgets return defaults unless overridden via `inputs` / `clicked`."""

    def __init__(self):
        super().__init__("streamlit")
        self.reset()

    def reset(self):
        self.session_state = {}
        self.inputs = {}       # key-or-label -> value returned by an input widget
        self.clicked = set()   # key-or-label of buttons that are "pressed" this run
        self.log = []          # (kind, text) for every message/markdown shown
        self.uploads = {}      # key -> fake uploaded file

    # --- helpers
    def _log(self, kind, text):
        self.log.append((kind, str(text)))

    def texts(self, kind=None):
        return [t for k, t in self.log if kind is None or k == kind]

    def _pick(self, key, label, default):
        for k in (key, label):
            if k is not None and k in self.inputs:
                return self.inputs[k]
        return default

    # --- config / control
    def set_page_config(self, **k): pass
    def rerun(self): raise Rerun()
    def cache_resource(self, f=None, **k):
        return f if f else (lambda g: g)

    # --- text output
    def title(self, t, **k): self._log("title", t)
    def header(self, t, **k): self._log("header", t)
    def subheader(self, t, **k): self._log("subheader", t)
    def caption(self, t, **k): self._log("caption", t)
    def write(self, *t, **k): self._log("write", " ".join(map(str, t)))
    def markdown(self, t, **k): self._log("markdown", t)
    def code(self, t, **k): self._log("code", t)
    def divider(self): pass
    def metric(self, label, value, **k): self._log("metric", f"{label}={value}")
    def success(self, t, **k): self._log("success", t)
    def error(self, t, **k): self._log("error", t)
    def warning(self, t, **k): self._log("warning", t)
    def info(self, t, **k): self._log("info", t)

    # --- layout
    def columns(self, n, **k):
        n = n if isinstance(n, int) else len(n)
        return [_Ctx() for _ in range(n)]
    def tabs(self, labels): return [_Ctx() for _ in labels]
    def container(self, **k): return _Ctx()
    def expander(self, *a, **k): return _Ctx()
    def spinner(self, *a, **k): return _Ctx()
    def form(self, *a, **k): return _Ctx()
    @property
    def sidebar(self): return _Ctx()

    # --- inputs
    def text_input(self, label, value="", key=None, **k): return self._pick(key, label, value)
    def text_area(self, label, value="", key=None, **k): return self._pick(key, label, value)
    def number_input(self, label, value=0, key=None, **k): return self._pick(key, label, value)
    def checkbox(self, label, value=False, key=None, **k): return self._pick(key, label, value)
    def date_input(self, label, value=None, key=None, **k): return self._pick(key, label, value)
    def selectbox(self, label, options, index=0, key=None, format_func=None, **k):
        options = list(options)
        return self._pick(key, label, options[index] if options else None)
    def multiselect(self, label, options=(), default=None, key=None, **k):
        return self._pick(key, label, list(default or []))
    def file_uploader(self, label, key=None, **k): return self.uploads.get(key)

    def button(self, label, key=None, **k): return (key in self.clicked) or (label in self.clicked)
    form_submit_button = button


def install_streamlit():
    fake = FakeStreamlit()
    sys.modules["streamlit"] = fake
    return fake


# ------------------------------------------------------------------ faiss (numpy)
class _IndexFlatIP:
    def __init__(self, d):
        self.d, self.vectors = d, np.empty((0, d), dtype=np.float32)

    @property
    def ntotal(self): return len(self.vectors)

    def add(self, x): self.vectors = np.vstack([self.vectors, np.asarray(x, dtype=np.float32)])

    def search(self, q, k):
        scores = q @ self.vectors.T
        order = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, order, axis=1), order

    def reconstruct_n(self, start, n): return self.vectors[start:start + n].copy()


def install_faiss():
    mod = types.ModuleType("faiss")
    mod.IndexFlatIP = _IndexFlatIP
    def normalize_L2(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1
        x /= norms
    mod.normalize_L2 = normalize_L2
    def write_index(index, path): np.save(path + ".npy", index.vectors); open(path, "w").write("idx")
    def read_index(path):
        vectors = np.load(path + ".npy")
        index = _IndexFlatIP(vectors.shape[1]); index.add(vectors); return index
    mod.write_index, mod.read_index = write_index, read_index
    sys.modules["faiss"] = mod


# ------------------------------------------------------------------ sentence_transformers (hashing bag of words)
class _FakeModel:
    def __init__(self, *a, **k): pass

    def get_sentence_embedding_dimension(self): return 384

    def _one(self, text):
        v = np.zeros(384, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 384] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts, **k):
        return np.stack([self._one(t) for t in texts]) if isinstance(texts, list) else self._one(texts)


def install_sentence_transformers():
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _FakeModel
    sys.modules["sentence_transformers"] = mod


# ------------------------------------------------------------------ fitz (PyMuPDF)
FAKE_PDF_PAGES = {}  # path -> [page text, ...]


class _Page:
    def __init__(self, t): self.t = t
    def get_text(self, mode="text"): return self.t


class _Doc:
    def __init__(self, pages): self.pages = pages
    def __iter__(self): return iter(_Page(t) for t in self.pages)
    def __len__(self): return len(self.pages)
    def close(self): pass


def install_fitz():
    mod = types.ModuleType("fitz")
    mod.open = lambda path: _Doc(FAKE_PDF_PAGES[str(path)])
    sys.modules["fitz"] = mod


def install_all():
    st = install_streamlit()
    install_faiss()
    install_sentence_transformers()
    install_fitz()
    return st
