import { useCallback, useEffect, useState } from 'react';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import FileUpload from './components/FileUpload';
import InputBar from './components/InputBar';
import { Button, Input, Modal } from 'antd';
import {
  createSession,
  deleteSession as apiDeleteSession,
  getSession,
  healthCheck,
} from './api';
import type { Message, SessionInfo } from './types';
import { useStream } from './hooks/useStream';
import testDocUrl from './components/assistant_test_doc.txt?url';

const SESSIONS_KEY = 'myrag_sessions';

export default function App() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [currentSid, setCurrentSid] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [brainInfo, setBrainInfo] = useState<SessionInfo | null>(null);
  const [suggestedQuestions, setSuggestedQuestions] = useState<string[]>([]);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [brainName, setBrainName] = useState('MyRAG Brain');
  const [creating, setCreating] = useState(false);
  const { streaming, start, stop } = useStream();

  useEffect(() => {
    const raw = localStorage.getItem(SESSIONS_KEY);
    if (raw) {
      const list: SessionInfo[] = JSON.parse(raw);
      setSessions(list);
      if (list.length > 0) setCurrentSid(list[0].session_id);
    }
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
  }, []);

  useEffect(() => {
    if (!currentSid) return;
    getSession(currentSid)
      .then(setBrainInfo)
      .catch(() => setBrainInfo(null));
    setMessages([]);
    setSuggestedQuestions([]);
  }, [currentSid]);

  const persistSessions = (list: SessionInfo[]) => {
    setSessions(list);
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(list));
  };

  const handleCreate = () => {
    setBrainName('MyRAG Brain');
    setCreateModalOpen(true);
  };

  const handleCreateSubmit = async () => {
    const name = brainName.trim();
    if (!name || creating) return;
    setCreating(true);
    try {
      const s = await createSession(name);
      persistSessions([s, ...sessions]);
      setCurrentSid(s.session_id);
      setCreateModalOpen(false);
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (sid: string) => {
    await apiDeleteSession(sid);
    persistSessions(sessions.filter((s) => s.session_id !== sid));
    if (currentSid === sid) setCurrentSid(null);
  };

  const handleUploaded = async (questions: string[]) => {
    if (!currentSid) return;
    const info = await getSession(currentSid);
    setBrainInfo(info);
    setSuggestedQuestions(questions);
    persistSessions(
      sessions.map((s) => (s.session_id === currentSid ? info : s)),
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
      await start(currentSid, question, (chunk) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === aiId ? { ...m, content: m.content + chunk } : m,
          ),
        );
      });
      setMessages((prev) =>
        prev.map((m) => (m.id === aiId ? { ...m, streaming: false } : m)),
      );
    },
    [currentSid, start],
  );

  return (
    <div className="h-full flex">
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
      </Modal>
      <main className="flex-1 flex flex-col bg-slate-50">
        {currentSid && brainInfo ? (
          <>
            <header className="bg-white border-b border-slate-200 px-6 py-3">
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
            </header>
            <FileUpload sessionId={currentSid} onUploaded={handleUploaded} />
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
