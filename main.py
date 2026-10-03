import asyncio
import json
import logging
import os
import traceback
from typing import List

import duckdb
import gradio as gr
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Logging (so errors appear in Vercel function logs) ─────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tgnew")

# ── Config ──────────────────────────────────────────────────────────────────
HF_DATASET  = os.environ.get("TG_DATASET", "Nischayydv/tgnew")
HF_FILE     = os.environ.get("TG_FILE", "indexed_telegram.parquet")
HF_REVISION = os.environ.get("TG_REVISION", "main")

PARQUET_URL = (
    f"https://huggingface.co/datasets/{HF_DATASET}"
    f"/resolve/{HF_REVISION}/{HF_FILE}"
)

SEARCH_FIELDS = ["user_id", "phone"]

# ── DuckDB setup ────────────────────────────────────────────────────────────
def _new_conn() -> duckdb.DuckDBPyConnection:
    """
    Create a fresh DuckDB connection configured for Vercel's sandbox.
    Every path points to /tmp because that's the only writable location.
    """
    os.makedirs("/tmp/duckdb_ext", exist_ok=True)
    os.makedirs("/tmp/duckdb_temp", exist_ok=True)

    con = duckdb.connect()

    # ── Point EVERYTHING to /tmp ────────────────────────────────────────
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_ext'")
    con.execute("SET temp_directory='/tmp/duckdb_temp'")
    con.execute("SET threads=2")
    con.execute("SET memory_limit='256MB'")

    # ── Install + load extensions ───────────────────────────────────────
    try:
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
    except Exception as e:
        logger.error(f"Failed to install/load httpfs: {e}")
        raise

    try:
        con.execute("INSTALL parquet;")
        con.execute("LOAD parquet;")
    except Exception as e:
        logger.error(f"Failed to install/load parquet: {e}")
        raise

    # ── Optional HF token ───────────────────────────────────────────────
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            con.execute(
                f"CREATE OR REPLACE SECRET hf "
                f"(TYPE huggingface, TOKEN '{token}')"
            )
        except Exception as e:
            logger.warning(f"HF secret creation failed: {e}")

    # ── Create the view ─────────────────────────────────────────────────
    con.execute(
        f"CREATE OR REPLACE VIEW people AS "
        f"SELECT * FROM read_parquet('{PARQUET_URL}')"
    )
    return con


# ── Helpers ─────────────────────────────────────────────────────────────────
def _escape(v: str) -> str:
    return v.replace("'", "''")


def _rows_to_dicts(con, rows) -> List[dict]:
    cols = [d[0] for d in con.description]
    return [dict(zip(cols, r)) for r in rows]


# ── Query functions (each opens its own connection) ─────────────────────────
def lookup_by_user_id(user_id: str, limit: int = 20) -> dict:
    uid = _escape(str(user_id).strip())
    con = _new_conn()
    try:
        sql = (
            f"SELECT user_id, phone, indexed_at FROM people "
            f"WHERE CAST(user_id AS VARCHAR) = '{uid}' LIMIT {limit}"
        )
        rows = _rows_to_dicts(con, con.execute(sql).fetchall())
    finally:
        con.close()

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
    con = _new_conn()
    try:
        sql = (
            f"SELECT user_id, phone, indexed_at FROM people "
            f"WHERE CAST(phone AS VARCHAR) = '{p}' LIMIT {limit}"
        )
        rows = _rows_to_dicts(con, con.execute(sql).fetchall())
    finally:
        con.close()

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
    con = _new_conn()
    try:
        where = (
            f"CAST(user_id AS VARCHAR) ILIKE '%{v}%' "
            f"OR CAST(phone AS VARCHAR) ILIKE '%{v}%'"
        )
        sql = f"SELECT user_id, phone, indexed_at FROM people WHERE {where} LIMIT {limit}"
        rows = _rows_to_dicts(con, con.execute(sql).fetchall())
    finally:
        con.close()
    return {"query": q, "count": len(rows), "results": rows}


def field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    v = _escape(value)
    con = _new_conn()
    try:
        if mode == "exact":
            sql = (f"SELECT user_id, phone, indexed_at FROM people "
                   f"WHERE CAST({field} AS VARCHAR) = '{v}' LIMIT {limit}")
        elif mode == "contains":
            v2 = v.replace("%", r"\%").replace("_", r"\_")
            sql = (f"SELECT user_id, phone, indexed_at FROM people "
                   f"WHERE CAST({field} AS VARCHAR) ILIKE '%{v2}%' ESCAPE '\\' LIMIT {limit}")
        else:
            raise ValueError(f"Unknown mode: {mode}")
        rows = _rows_to_dicts(con, con.execute(sql).fetchall())
    finally:
        con.close()
    return {"field": field, "value": value, "mode": mode,
            "count": len(rows), "results": rows}


# ── FastAPI ─────────────────────────────────────────────────────────────────
app = FastAPI(title="TGNew Search API")


# Global exception handler: log the real error + return it as JSON
@app.exception_handler(Exception)
async def _unhandled(request, exc):
    tb = traceback.format_exc()
    logger.error(f"Unhandled error on {request.url.path}:\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "traceback": tb},
    )


class BatchRequest(BaseModel):
    user_ids: List[str]
    limit: int = 20


@app.get("/")
def root():
    return {
        "app": "TGNew Search API",
        "dataset": HF_DATASET,
        "file": HF_FILE,
        "parquet_url": PARQUET_URL,
        "columns": ["user_id", "phone", "indexed_at"],
        "endpoints": {
            "user_id": "/user/{user_id}",
            "phone": "/phone/{phone}",
            "search": "/search?q=...",
            "field_search": "/search?q=...&field=user_id&mode=exact",
            "batch": "POST /users/batch",
            "health": "/health",
            "debug": "/debug",
            "ui": "/ui",
        },
        "docs": "/docs",
    }


@app.get("/health")
def health():
    try:
        con = _new_conn()
        n = con.execute("SELECT COUNT(*) FROM people").fetchone()[0]
        con.close()
        return {"status": "ok", "rows": n}
    except Exception as e:
        return {"status": "error", "detail": str(e), "traceback": traceback.format_exc()}


@app.get("/debug")
def debug():
    """
    Run this first if you get a 500. It returns the raw traceback
    from DuckDB connection setup, extension loading, and view creation.
    """
    out = {"parquet_url": PARQUET_URL}

    # Step 1: basic DuckDB
    try:
        con = duckdb.connect()
        out["duckdb_version"] = duckdb.__version__
        out["step1_connect"] = "ok"
    except Exception as e:
        out["step1_connect"] = f"FAIL: {e}"
        out["traceback"] = traceback.format_exc()
        return out

    # Step 2: extensions
    try:
        con.execute("SET home_directory='/tmp'")
        con.execute("SET extension_directory='/tmp/duckdb_ext'")
        con.execute("SET temp_directory='/tmp/duckdb_temp'")
        os.makedirs("/tmp/duckdb_ext", exist_ok=True)
        os.makedirs("/tmp/duckdb_temp", exist_ok=True)
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
        out["step2_httpfs"] = "ok"
    except Exception as e:
        out["step2_httpfs"] = f"FAIL: {e}"
        out["traceback"] = traceback.format_exc()
        con.close()
        return out

    # Step 3: read remote parquet
    try:
        n = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{PARQUET_URL}')"
        ).fetchone()[0]
        out["step3_read_parquet"] = f"ok — {n} rows"
    except Exception as e:
        out["step3_read_parquet"] = f"FAIL: {e}"
        out["traceback"] = traceback.format_exc()
        con.close()
        return out

    con.close()
    out["status"] = "ALL OK"
    return out


@app.get("/user/{user_id}")
async def user_lookup(
    user_id: str,
    limit: int = Query(20, ge=1, le=200),
    pretty: bool = Query(True),
):
    if not user_id.strip():
        raise HTTPException(422, "user_id cannot be empty")
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, lookup_by_user_id, user_id, limit)
    result = {"success": bool(data["count"]), **data}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


@app.get("/phone/{phone}")
async def phone_lookup(
    phone: str,
    limit: int = Query(20, ge=1, le=200),
    pretty: bool = Query(True),
):
    if not phone.strip():
        raise HTTPException(422, "phone cannot be empty")
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, lookup_by_phone, phone, limit)
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
        data = await loop.run_in_executor(None, field_search, field, q.strip(), mode, limit)
    else:
        data = await loop.run_in_executor(None, unified_search, q.strip(), limit)
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
        loop.run_in_executor(None, lookup_by_user_id, uid, req.limit)
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
                q_in = gr.Textbox(label="Search Query",
                                  placeholder="e.g. 989301897477 or 263493087", lines=1)
            with gr.Column(scale=1):
                limit_slider = gr.Slider(minimum=1, maximum=100, value=20, step=1,
                                         label="Max Results")
        btn = gr.Button("🔍 Search", variant="primary", size="lg")
        out = gr.Markdown(label="Results")
        btn.click(fn=ui_search, inputs=[q_in, limit_slider], outputs=out)
        q_in.submit(fn=ui_search, inputs=[q_in, limit_slider], outputs=out)
    return demo


gr.mount_gradio_app(app, build_ui(), path="/ui")
