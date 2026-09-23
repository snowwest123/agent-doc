"""SQL 查询工具 v2：支持 schema 感知。"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from myrag.tools.base import ToolWrapper

@tool
def get_db_schema(db_path: str) -> str:
    """查看数据库的所有表结构。"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        schema_text = ""
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            cols = cursor.fetchall()
            schema_text += f"\n## 表 {table}\n"
            for col in cols:
                schema_text += f"  - {col[1]} ({col[2]})\n"

 # 加上数据量统计
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            cnt = cursor.fetchone()[0]
            schema_text += f"  行数: {cnt}\n"

        conn.close()
        return schema_text or "空数据库"
    except Exception as e:
        return f"读取 schema 错误：{e}"

@tool
def sql_query(db_path: str, query: str) -> str:
    """执行 SQL 查询（SELECT only）。"""
    if not query.strip().upper().startswith("SELECT"):
        return "错误：只允许 SELECT 查询"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description] if cursor.description else []
        conn.close()

        if not rows:
            return "查询成功，无返回结果"
        result = " | ".join(columns) + "\n" + "-" * 40 + "\n"
        for row in rows[:30]:
            result += " | ".join(str(v) for v in row) + "\n"
        if len(rows) > 30:
            result += f"... 共 {len(rows)} 行"
        return result
    except Exception as e:
        return f"SQL 错误：{e}"

def create_sql_tools() -> tuple[ToolWrapper, ToolWrapper]:
    """返回 (schema tool, query tool)。"""
    schema_tool = ToolWrapper(
        tool=get_db_schema,
        name="get_db_schema",
        description="查看数据库的所有表结构，输入 db_path",
    )
    query_tool = ToolWrapper(
        tool=sql_query,
        name="sql_query",
        description="执行 SELECT 查询，输入 db_path 和 query",
    )
    return schema_tool, query_tool


# ===== 阿里云 Hologres 支持 =====

import os


def _get_hologres_backend():
    """从环境变量构造 Hologres 连接。"""
    import psycopg

    host = os.getenv("HOLOGRES_HOST")
    if not host:
        raise ValueError("需要设置 HOLOGRES_HOST 等环境变量")
    return psycopg.connect(
        host=host,
        port=int(os.getenv("HOLOGRES_PORT", "80")),
        dbname=os.getenv("HOLOGRES_DB", ""),
        user=os.getenv("HOLOGRES_USER", ""),
        password=os.getenv("HOLOGRES_PASSWORD", ""),
        connect_timeout=10,
    )


@tool
def get_hologres_schema() -> str:
    """查看阿里云 Hologres 数据库的所有表结构。从环境变量自动读连接信息。"""
    try:
        conn = _get_hologres_backend()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position
            """)
            rows = cur.fetchall()

        schema_dict = {}
        for table, col, dtype in rows:
            schema_dict.setdefault(table, []).append((col, dtype))

        schema_text = ""
        for table, cols in schema_dict.items():
            schema_text += f"\n## 表 {table}\n"
            for col_name, col_type in cols:
                schema_text += f"  - {col_name}: {col_type}\n"
            with conn.cursor() as cur:
                cur.execute(f'SELECT COUNT(*) FROM "{table}"')
                cnt = cur.fetchone()[0]
            schema_text += f"  行数: {cnt}\n"
        conn.close()
        return schema_text or "空数据库"
    except Exception as e:
        return f"读取 schema 错误：{e}"


@tool
def hologres_query(query: str) -> str:
    """在阿里云 Hologres 上执行 SELECT 查询。从环境变量自动读连接信息。"""
    if not query.strip().upper().startswith("SELECT"):
        return "错误：只允许 SELECT 查询"
    try:
        conn = _get_hologres_backend()
        with conn.cursor() as cur:
            cur.execute(query)
            columns = [d.name for d in cur.description] if cur.description else []
            rows = cur.fetchall()
        conn.close()

        if not rows:
            return "查询成功，无返回结果"
        result = " | ".join(columns) + "\n" + "-" * 40 + "\n"
        for row in rows[:30]:
            result += " | ".join(str(v) for v in row) + "\n"
        if len(rows) > 30:
            result += f"... 共 {len(rows)} 行"
        return result
    except Exception as e:
        return f"SQL 错误：{e}"


def create_hologres_tools() -> tuple[ToolWrapper, ToolWrapper]:
    """创建 Hologres 工具（自动从环境变量读连接信息）。"""
    schema_tool = ToolWrapper(
        tool=get_hologres_schema,
        name="get_hologres_schema",
        description="查看阿里云 Hologres 数据库的所有表结构",
    )
    query_tool = ToolWrapper(
        tool=hologres_query,
        name="hologres_query",
        description="在阿里云 Hologres 上执行 SELECT 查询，输入参数 query",
    )
    return schema_tool, query_tool