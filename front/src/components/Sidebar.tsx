import { Plus, Trash2, MessageSquare, FileText } from 'lucide-react'
import type { SessionInfo } from '../types'

interface Props {
  sessions: SessionInfo[]
  currentSid: string | null
  onSelect: (sid: string) => void
  onCreate: () => void
  onDelete: (sid: string) => void
  backendOk: boolean | null
}

export default function Sidebar({
  sessions,
  currentSid,
  onSelect,
  onCreate,
  onDelete,
  backendOk,
}: Props) {
  return (
    <aside className="w-72 bg-slate-900 text-slate-100 flex flex-col">
      <div className="p-4 border-b border-slate-700">
        <div className="flex items-center gap-2 mb-3">
          <MessageSquare className="w-6 h-6 text-primary-500" />
          <h1 className="text-lg font-bold">MyRAG</h1>
        </div>
        <button
          onClick={onCreate}
          className="w-full flex items-center justify-center gap-2 bg-primary-600 hover:bg-primary-700 transition rounded-lg py-2 text-sm font-medium"
        >
          <Plus className="w-4 h-4" /> 新建会话
        </button>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin p-2">
        {sessions.length === 0 && (
          <p className="text-xs text-slate-500 text-center mt-8 px-2">
            还没有会话，点击上方按钮创建一个
          </p>
        )}
        {sessions.map((s) => (
          <div
            key={s.session_id}
            className={`group flex items-center justify-between rounded-lg px-3 py-2 mb-1 cursor-pointer transition ${
              currentSid === s.session_id
                ? 'bg-primary-600 text-white'
                : 'hover:bg-slate-800 text-slate-300'
            }`}
            onClick={() => onSelect(s.session_id)}
          >
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">{s.brain_name}</p>
              <p className="text-xs opacity-70 flex items-center gap-1">
                <FileText className="w-3 h-3" /> {s.nb_chunks} chunks
              </p>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation()
                if (confirm('删除该会话？')) onDelete(s.session_id)
              }}
              className="opacity-0 group-hover:opacity-100 hover:text-red-400 transition"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        ))}
      </div>

      <div className="p-3 border-t border-slate-700 text-xs">
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              backendOk === null
                ? 'bg-slate-500'
                : backendOk
                  ? 'bg-green-500'
                  : 'bg-red-500'
            }`}
          />
          <span className="text-slate-400">
            {backendOk === null
              ? '检查中...'
              : backendOk
                ? '后端已连接'
                : '后端断开'}
          </span>
        </div>
      </div>
    </aside>
  )
}