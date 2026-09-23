# ===== 前端构建阶段：React/Vite -> 静态文件 =====
ARG NODE_IMAGE=docker.m.daocloud.io/library/node:20-alpine
ARG PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.11-slim
ARG NGINX_IMAGE=docker.m.daocloud.io/library/nginx:alpine

FROM ${NODE_IMAGE} AS frontend-builder

WORKDIR /front

# 先复制依赖清单，利用 Docker 缓存
COPY front/package.json ./
RUN npm config set registry https://registry.npmmirror.com && \
    npm install --no-audit --no-fund

COPY front/ ./
# 生产环境前端通过同源 nginx 的 /api 访问后端
ARG VITE_API_BASE=/api
ENV VITE_API_BASE=${VITE_API_BASE}
RUN npm run build


# ===== 后端构建阶段：FastAPI API =====
FROM ${PYTHON_IMAGE} AS api

# 设置工作目录
WORKDIR /app

# 系统依赖（psycopg 需要 libpq；faiss-cpu 需要 libgomp1；healthcheck 需要 curl）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件（利用 Docker 缓存）
COPY pyproject.toml ./

# 装包
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
pip config set global.timeout 120 && \
pip install --no-cache-dir --upgrade pip && \
pip install --no-cache-dir --timeout 120 --retries 5 \
    "fastapi" "uvicorn[standard]" "python-multipart" \
    "psycopg[binary]>=0.3,<0.4" "langchain-core>=0.3,<0.4" "langchain-openai>=0.3,<0.4" \
    "langchain-community" "langgraph" "duckdb" "tiktoken" "pydantic>=2.8" \
    "pyyaml" "dashscope" "faiss-cpu" "rich" "python-dotenv" "nest_asyncio" "redis>=5.0"

# 复制项目代码
COPY myrag ./myrag/
COPY examples/step13_fastapi ./app/
# 注意：.env 不打进镜像，由 docker-compose.yml 的 env_file 在运行时注入

# 暴露端口
EXPOSE 8000

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# 启动
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]


# ===== 前端运行阶段：Nginx 服务 React 静态文件 + 反代 API =====
FROM ${NGINX_IMAGE} AS frontend-runtime

COPY --from=frontend-builder /front/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
