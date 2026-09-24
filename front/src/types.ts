// 会话工作模式：knowledge=文档问答 / database=查业务数据
export type SessionMode = 'knowledge' | 'database'

export interface SessionInfo {
  session_id: string
  brain_name: string
  nb_chunks: number
  files: string[]
  created_at: string
  mode: SessionMode
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  streaming?: boolean
}

export interface BrainInfo {
  session_id: string
  name: string
  llm_model: string
  nb_chunks: number
  uploaded_files: string[]
}