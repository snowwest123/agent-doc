import { useCallback, useEffect, useState } from 'react';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import FileUpload from './components/FileUpload';
import InputBar from './components/InputBar';
import { Button, Input, Modal, Radio, Spin, Tag, Tooltip } from 'antd';
import {
  askSqlStream,
  createSession,
  deleteSession as apiDeleteSession,
  getHistory,
  getSession,
  healthCheck,
  listSessions,
  setSessionMode,
} from './api';
import type { Message, SessionInfo, SessionMode } from './types';
import { useStream } from './hooks/useStream';
import testDocUrl from './components/assistant_test_doc.txt?url';

const CURRENT_SID_KEY = 'myrag_current_sid';

export default function App() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [currentSid, setCurrentSid] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [brainInfo, setBrainInfo] = useState<SessionInfo | null>(null);
  const [suggestedQuestions, setSuggestedQuestions] = useState<string[]>([]);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [brainName, setBrainName] = useState('MyRAG Brain');
  const [createMode, setCreateMode] = useState<SessionMode>('knowledge');
  const [creating, setCreating] = useState(false);
  const [modeSwitching, setModeSwitching] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const { streaming, start, stop } = useStream();

  // 从后端拉全量会话（Hologres 持久化），刷新也不会丢
  const refreshSessions = useCallback(async () => {
    setInitialLoading(true);
    try {
      const list = await listSessions();
      setSessions(list);
      const saved = localStorage.getItem(CURRENT_SID_KEY);
      if (saved && list.some((s) => s.session_id === saved)) {
        setCurrentSid(saved);
      } else if (list.length > 0) {
        setCurrentSid(list[0].session_id);
      } else {
        setCurrentSid(null);
      }
    } catch (err) {
      console.error('listSessions failed', err);
      setInitialLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshSessions();
    const tick = async () => {
      try {
        const h = await healthCheck();
        setBackendOk(h.status === 'ok' && h.redis === 'ok');
      } catch {
        setBackendOk(false);
      }
    };
    tick();
    const t = setInterval(tick, 10000);
    return () => clearInterval(t);
  }, [refreshSessions]);

  // 切换 / 初次加载 currentSid 时：拉脑信息 + 聊天历史
  useEffect(() => {
    if (!currentSid) {
      setBrainInfo(null);
      setMessages([]);
      setSuggestedQuestions([]);
      setInitialLoading(false);
      return;
    }
    localStorage.setItem(CURRENT_SID_KEY, currentSid);
    let cancelled = false;
    const sessionRequest = getSession(currentSid)
      .then((info) => {
        if (!cancelled) setBrainInfo(info);
      })
      .catch(() => {
        if (!cancelled) setBrainInfo(null);
      });
    const historyRequest = getHistory(currentSid)
      .then((history) => {
        if (cancelled) return;
        setMessages(
          history.map((m, idx) => ({
            id: `r-${currentSid}-${idx}`,
            role: m.role,
            content: m.content,
          })),
        );
      })
      .catch(() => {
        if (!cancelled) setMessages([]);
      });
    void Promise.all([sessionRequest, historyRequest]).finally(() => {
      if (!cancelled) setInitialLoading(false);
    });
    setSuggestedQuestions([]);
    return () => {
      cancelled = true;
    };
  }, [currentSid]);

  const handleCreate = () => {
    setBrainName('MyRAG Brain');
    setCreateMode('knowledge');
    setCreateModalOpen(true);
  };

  const handleCreateSubmit = async () => {
    const name = brainName.trim();
    if (!name || creating) return;
    setCreating(true);
    try {
      const s = await createSession(name, createMode);
      setSessions((prev) => [
        s,
        ...prev.filter((x) => x.session_id !== s.session_id),
      ]);
      setCurrentSid(s.session_id);
      setCreateModalOpen(false);
    } finally {
      setCreating(false);
    }
  };

  const handleModeChange = async (next: SessionMode) => {
    if (!currentSid || !brainInfo || brainInfo.mode === next || modeSwitching)
      return;
    stop();
    setMessages([]);
    setSuggestedQuestions([]);
    setModeSwitching(true);
    try {
      const updated = await setSessionMode(currentSid, next);
      setBrainInfo(updated);
      setSessions((prev) =>
        prev.map((s) => (s.session_id === currentSid ? updated : s)),
      );
    } finally {
      setModeSwitching(false);
    }
  };

  const handleDelete = async (sid: string) => {
    await apiDeleteSession(sid);
    setSessions((prev) => prev.filter((s) => s.session_id !== sid));
    if (currentSid === sid) {
      setCurrentSid(null);
      setMessages([]);
      setBrainInfo(null);
    }
  };

  const handleUploaded = async (questions: string[]) => {
    if (!currentSid) return;
    const info = await getSession(currentSid);
    setBrainInfo(info);
    setSuggestedQuestions(questions);
    setSessions((prev) =>
      prev.map((s) => (s.session_id === currentSid ? info : s)),
    );
  };

  const handleSend = useCallback(
    async (question: string) => {
      if (!currentSid) return;
      const userMsg: Message = {
        id: `u-${Date.now()}`,
        role: 'user',
        content: question,
      };
      const aiId = `a-${Date.now()}`;
      const aiMsg: Message = {
        id: aiId,
        role: 'assistant',
        content: '',
        streaming: true,
      };
      setMessages((prev) => [...prev, userMsg, aiMsg]);
      if (brainInfo?.mode === 'knowledge') {
        await start(currentSid, question, (chunk) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === aiId ? { ...m, content: m.content + chunk } : m,
            ),
          );
        });
      } else if (brainInfo?.mode === 'database') {
        await askSqlStream(currentSid, question, (event) => {
          if (event.type === 'llm_start') {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === aiId
                  ? { ...m, content: `${m.content}${event.message}\n` }
                  : m,
              ),
            );
          } else if (event.type === 'llm_decision') {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === aiId
                  ? {
                      ...m,
                      content: `${m.content}Agent 决策：${event.action}\n`,
                    }
                  : m,
              ),
            );
          } else if (event.type === 'tool_call') {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === aiId
                  ? {
                      ...m,
                      content: `${m.content}正在调用工具：${event.tool}\n`,
                    }
                  : m,
              ),
            );
          } else if (event.type === 'tool_result') {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === aiId
                  ? {
                      ...m,
                      content: `${m.content}\n工具结果：\n${event.result}`,
                    }
                  : m,
              ),
            );
          } else if (event.type === 'final_answer') {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === aiId
                  ? {
                      ...m,
                      content: `${m.content}\n最终答案：\n${event.answer}`,
                    }
                  : m,
              ),
            );
          }
        });
      }
      setMessages((prev) =>
        prev.map((m) => (m.id === aiId ? { ...m, streaming: false } : m)),
      );
    },
    [brainInfo?.mode, currentSid, start, askSqlStream],
  );

  return (
    <div className="relative h-full flex">
      {(initialLoading || modeSwitching) && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-white/70">
          <Spin
            size="large"
            tip={initialLoading ? '正在加载会话历史...' : '正在切换会话模式...'}
          />
        </div>
      )}
      <Sidebar
        sessions={sessions}
        currentSid={currentSid}
        onSelect={setCurrentSid}
        onCreate={handleCreate}
        onDelete={handleDelete}
        backendOk={backendOk}
      />
      <Modal
        title="创建 Brain"
        open={createModalOpen}
        onOk={handleCreateSubmit}
        onCancel={() => setCreateModalOpen(false)}
        okText="创建"
        cancelText="取消"
        confirmLoading={creating}
        destroyOnClose
      >
        <Input
          autoFocus
          value={brainName}
          placeholder="请输入 Brain 名称"
          maxLength={50}
          showCount
          onChange={(event) => setBrainName(event.target.value)}
          onPressEnter={handleCreateSubmit}
        />
        <div className="mt-4">
          <p className="text-sm text-slate-600 mb-2">工作模式</p>
          <Radio.Group
            value={createMode}
            onChange={(e) => setCreateMode(e.target.value as SessionMode)}
          >
            <Tooltip title="上传文档 → RAG 问答（已上线）">
              <Radio.Button value="knowledge">文档问答</Radio.Button>
            </Tooltip>
            <Tooltip title="查询 Hologres 业务数据">
              <Radio.Button value="database">数据库查询</Radio.Button>
            </Tooltip>
          </Radio.Group>
          <p className="mt-2 text-xs text-slate-400">
            模式可在创建后在头部随时切换
          </p>
        </div>
      </Modal>
      <main className="flex-1 flex flex-col bg-slate-50">
        {currentSid && brainInfo ? (
          <>
            <header className="bg-white border-b border-slate-200 px-6 py-3 flex items-start justify-between gap-4">
              <div>
                <h2 className="font-semibold text-slate-800">
                  {brainInfo.brain_name}
                </h2>
                <p className="text-xs text-slate-500">
                  {brainInfo.nb_chunks} chunks · {brainInfo.files.length} files
                </p>
                <p className="mt-1 text-xs text-slate-400">
                  支持格式：TXT 单个文件最大 10 MB
                </p>
                <Button
                  type="link"
                  href={testDocUrl}
                  download="assistant_test_doc.txt"
                  className="h-auto p-0 text-xs"
                >
                  下载测试文件（方便上传知识库）
                </Button>
              </div>
              <div className="shrink-0">
                <p className="text-xs text-slate-500 mb-1 text-right">
                  工作模式
                </p>
                <Radio.Group
                  size="small"
                  value={brainInfo.mode}
                  onChange={(e) =>
                    handleModeChange(e.target.value as SessionMode)
                  }
                >
                  <Radio.Button value="knowledge">文档</Radio.Button>
                  <Radio.Button value="database">数据库</Radio.Button>
                </Radio.Group>
                {brainInfo.mode !== 'knowledge' && (
                  <Tag color="orange" className="mt-1 block text-center">
                    当前走 SQL 问答端点（您可以问b）
                  </Tag>
                )}
              </div>
            </header>
            {brainInfo.mode === 'knowledge' && (
              <FileUpload sessionId={currentSid} onUploaded={handleUploaded} />
            )}
            <ChatWindow
              messages={messages}
              streaming={streaming}
              suggestedQuestions={suggestedQuestions}
              onSelectQuestion={handleSend}
            />
            <InputBar onSend={handleSend} streaming={streaming} onStop={stop} />
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-slate-400">
            请先创建会话
          </div>
        )}
      </main>
    </div>
  );
}
