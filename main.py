import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List

import duckdb
import gradio as gr
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────
HF_DATASET  = os.environ.get("TG_DATASET", "Nischayydv/tgnew")
HF_FILE     = os.environ.get("TG_FILE", "indexed_telegram.parquet")
HF_REVISION = os.environ.get("TG_REVISION", "main")

PARQUET_URL = (
    f"https://huggingface.co/datasets/{HF_DATASET}"
    f"/resolve/{HF_REVISION}/{HF_FILE}"
)

SEARCH_FIELDS = ["user_id", "phone"]

PARALLELISM      = int(os.environ.get("TG_PARALLEL", "2"))
THREADS_PER_CONN = int(os.environ.get("TG_THREADS_PER_CONN", "2"))

# ── DuckDB Connection Pool ──────────────────────────────────────────────────
_conns: List[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")


def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute(f"SET threads = {THREADS_PER_CONN}")

    token = os.environ.get("HF_TOKEN")
    if token:
        con.execute(
            f"CREATE OR REPLACE SECRET hf "
            f"(TYPE huggingface, TOKEN '{token}')"
        )

    con.execute(
        f"CREATE OR REPLACE VIEW people AS "
        f"SELECT * FROM read_parquet('{PARQUET_URL}')"
    )
    return con


def _thread_id() -> int:
    tid = getattr(_thread_local, "id", None)
    if tid is None:
        with _conns_lock:
            tid = len(_conns)
            _thread_local.id = tid
    return tid


def _get_conn() -> duckdb.DuckDBPyConnection:
    ident = _thread_id()
    with _conns_lock:
        while len(_conns) <= ident:
            _conns.append(_new_conn())
    return _conns[ident]


# ── Helpers ─────────────────────────────────────────────────────────────────
def _escape(v: str) -> str:
    return v.replace("'", "''")


def _rows_to_dicts(con, rows) -> List[dict]:
    cols = [d[0] for d in con.description]
    return [dict(zip(cols, r)) for r in rows]


# ── Core: user_id → phone ───────────────────────────────────────────────────
def lookup_by_user_id(user_id: str, limit: int = 20) -> dict:
    uid = _escape(str(user_id).strip())
    sql = (
        f"SELECT user_id, phone, indexed_at FROM people "
        f"WHERE CAST(user_id AS VARCHAR) = '{uid}' LIMIT {limit}"
    )
    con = _get_conn()
    rows = _rows_to_dicts(con, con.execute(sql).fetchall())

    phones, seen = [], set()
    for r in rows:
        p = r.get("phone")
        if p and p not in seen:
            seen.add(p)
            phones.append(p)

    return {
        "user_id": user_id,
        "phone_numbers": phones,
        "count": len(rows),
        "results": rows,
    }


def lookup_by_phone(phone: str, limit: int = 20) -> dict:
    p = _escape(str(phone).strip())
    sql = (
        f"SELECT user_id, phone, indexed_at FROM people "
        f"WHERE CAST(phone AS VARCHAR) = '{p}' LIMIT {limit}"
    )
    con = _get_conn()
    rows = _rows_to_dicts(con, con.execute(sql).fetchall())

    user_ids, seen = [], set()
    for r in rows:
        u = r.get("user_id")
        if u and u not in seen:
            seen.add(u)
            user_ids.append(u)

    return {
        "phone": phone,
        "user_ids": user_ids,
        "count": len(rows),
        "results": rows,
    }


def unified_search(q: str, limit: int = 20) -> dict:
    q = q.strip()
    if not q:
        return {"query": q, "count": 0, "results": []}
    v = _escape(q)
    where = (
        f"CAST(user_id AS VARCHAR) ILIKE '%{v}%' "
        f"OR CAST(phone AS VARCHAR) ILIKE '%{v}%'"
    )
    sql = f"SELECT user_id, phone, indexed_at FROM people WHERE {where} LIMIT {limit}"
    con = _get_conn()
    rows = _rows_to_dicts(con, con.execute(sql).fetchall())
    return {"query": q, "count": len(rows), "results": rows}


def field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    v = _escape(value)

    if mode == "exact":
        sql = (f"SELECT user_id, phone, indexed_at FROM people "
               f"WHERE CAST({field} AS VARCHAR) = '{v}' LIMIT {limit}")
    elif mode == "contains":
        v2 = v.replace("%", r"\%").replace("_", r"\_")
        sql = (f"SELECT user_id, phone, indexed_at FROM people "
               f"WHERE CAST({field} AS VARCHAR) ILIKE '%{v2}%' ESCAPE '\\' LIMIT {limit}")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    con = _get_conn()
    rows = _rows_to_dicts(con, con.execute(sql).fetchall())
    return {"field": field, "value": value, "mode": mode,
            "count": len(rows), "results": rows}


# ── FastAPI ─────────────────────────────────────────────────────────────────
app = FastAPI(title="TGNew Search API")


class BatchRequest(BaseModel):
    user_ids: List[str]
    limit: int = 20


@app.get("/")
def root():
    return {
        "app": "TGNew Search API",
        "dataset": HF_DATASET,
        "file": HF_FILE,
        "columns": ["user_id", "phone", "indexed_at"],
        "endpoints": {
            "user_id": "/user/{user_id}",
            "phone": "/phone/{phone}",
            "search": "/search?q=...",
            "field_search": "/search?q=...&field=user_id&mode=exact",
            "batch": "POST /users/batch",
            "health": "/health",
            "ui": "/ui",
        },
        "docs": "/docs",
    }


@app.get("/health")
def health():
    try:
        con = _get_conn()
        n = con.execute("SELECT COUNT(*) FROM people").fetchone()[0]
        return {"status": "ok", "rows": n}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ⭐ user_id → phone
@app.get("/user/{user_id}")
async def user_lookup(
    user_id: str,
    limit: int = Query(20, ge=1, le=200),
    pretty: bool = Query(True),
):
    if not user_id.strip():
        raise HTTPException(422, "user_id cannot be empty")
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(pool, lookup_by_user_id, user_id, limit)
    result = {"success": bool(data["count"]), **data}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


# ⭐ phone → user_id
@app.get("/phone/{phone}")
async def phone_lookup(
    phone: str,
    limit: int = Query(20, ge=1, le=200),
    pretty: bool = Query(True),
):
    if not phone.strip():
        raise HTTPException(422, "phone cannot be empty")
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(pool, lookup_by_phone, phone, limit)
    result = {"success": bool(data["count"]), **data}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


@app.get("/search")
async def search(
    q: str | None = Query(None),
    field: str | None = Query(None),
    mode: str = Query("contains"),
    limit: int = Query(20, ge=1, le=200),
    pretty: bool = Query(True),
):
    if not q or not q.strip():
        raise HTTPException(422, "Provide q")
    loop = asyncio.get_running_loop()
    if field:
        data = await loop.run_in_executor(
            pool, field_search, field, q.strip(), mode, limit
        )
    else:
        data = await loop.run_in_executor(pool, unified_search, q.strip(), limit)
    result = {"success": bool(data["count"]), **data, "total": data["count"]}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


@app.post("/users/batch")
async def users_batch(req: BatchRequest):
    if not req.user_ids:
        raise HTTPException(400, "user_ids must not be empty")
    if len(req.user_ids) > 100:
        raise HTTPException(400, "max 100 user_ids per batch")
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(pool, lookup_by_user_id, uid, req.limit)
        for uid in req.user_ids
    ]
    results = await asyncio.gather(*tasks)
    return Response(
        content=json.dumps({"count": len(results), "results": list(results)},
                           indent=2, ensure_ascii=False),
        media_type="application/json",
    )


# ── Gradio UI ───────────────────────────────────────────────────────────────
def ui_search(query: str, limit: int) -> str:
    if not query or not query.strip():
        return "⚠️ Enter a user_id or phone number."
    q = query.strip()
    try:
        data = unified_search(q, int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"

    if not data["count"]:
        return f"🔍 **Query:** `{q}`\n\n❌ **No records found.**"

    lines = [f"🔍 **Query:** `{q}`  |  **Found:** {data['count']}", "", "---", ""]
    for i, r in enumerate(data["results"], 1):
        lines.append(f"### Result {i}")
        for k in ["user_id", "phone", "indexed_at"]:
            if r.get(k):
                lines.append(f"**{k}:** {r[k]}")
        lines.append("")
    return "\n\n".join(lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="TGNew Search", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 TGNew Search")
        gr.Markdown("Search by **user_id** or **phone**")

        with gr.Row():
            with gr.Column(scale=3):
                q_in = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g. 989301897477 or 263493087",
                    lines=1,
                )
            with gr.Column(scale=1):
                limit_slider = gr.Slider(
                    minimum=1, maximum=100, value=20, step=1,
                    label="Max Results",
                )

        btn = gr.Button("🔍 Search", variant="primary", size="lg")
        out = gr.Markdown(label="Results")

        btn.click(fn=ui_search, inputs=[q_in, limit_slider], outputs=out)
        q_in.submit(fn=ui_search, inputs=[q_in, limit_slider], outputs=out)
    return demo


gr.mount_gradio_app(app, build_ui(), path="/ui")
