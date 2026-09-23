import { useState } from 'react'
import { Send, Square } from 'lucide-react'

interface Props {
  onSend: (text: string) => void
  streaming: boolean
  onStop: () => void
  disabled?: boolean
}

export default function InputBar({
  onSend,
  streaming,
  onStop,
  disabled,
}: Props) {
  const [text, setText] = useState('')

  function submit() {
    const q = text.trim()
    if (!q || streaming) return
    onSend(q)
    setText('')
  }

  return (
    <div className="bg-white border-t border-slate-200 p-4">
      <div className="flex items-end gap-2 max-w-4xl mx-auto">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
          }}
          placeholder={disabled ? '请先创建会话并上传文档' : '输入问题，回车发送'}
          disabled={disabled}
          rows={1}
          className="flex-1 resize-none rounded-lg border border-slate-300 px-4 py-2 text-sm focus:outline-none focus:border-primary-500 disabled:bg-slate-50 disabled:text-slate-400"
        />
        {streaming ? (
          <button
            onClick={onStop}
            className="flex items-center gap-1 px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg text-sm"
          >
            <Square className="w-4 h-4" /> 停止
          </button>
        ) : (
          <button
            onClick={submit}
            disabled={disabled || !text.trim()}
            className="flex items-center gap-1 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg text-sm disabled:opacity-50"
          >
            <Send className="w-4 h-4" /> 发送
          </button>
        )}
      </div>
    </div>
  )
}