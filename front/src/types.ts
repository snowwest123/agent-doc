export interface SessionInfo {
  session_id: string
  brain_name: string
  nb_chunks: number
  files: string[]
  created_at: string
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