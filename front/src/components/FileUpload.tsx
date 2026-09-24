import { useRef, useState } from 'react';
import { Spin } from 'antd';
import { Upload, FileText, X } from 'lucide-react';
import { getSuggestedQuestions, uploadFiles } from '../api';

const MAX_FILE_SIZE = 10 * 1024 * 1024;

interface Props {
  sessionId: string;
  onUploaded: (questions: string[]) => void;
}

export default function FileUpload({ sessionId, onUploaded }: Props) {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleUpload(selectedFiles: File[]) {
    if (selectedFiles.length === 0) return;
    setUploading(true);
    try {
      await uploadFiles(sessionId, selectedFiles);
      const questions = await getSuggestedQuestions(sessionId);
      setFiles([]);
      onUploaded(questions);
    } catch (err) {
      alert(`上传失败：${(err as Error).message}`);
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="relative bg-white border-b border-slate-200 px-6 py-3">
      {uploading && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-white/70">
          <Spin size="large" tip="正在上传并处理文件..." />
        </div>
      )}
      <div className="flex items-center gap-2 flex-wrap">
        <button
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
          className="flex items-center gap-1 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-md text-sm disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Upload className="w-4 h-4" /> 请选择文件
        </button>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".txt"
          hidden
          onChange={(e) => {
            const selectedFiles = Array.from(e.target.files || []);
            const invalidFile = selectedFiles.find(
              (file) => file.size > MAX_FILE_SIZE,
            );
            if (invalidFile) {
              alert(`文件 ${invalidFile.name} 超过 10 MB 限制，请重新选择`);
              e.target.value = '';
              return;
            }
            setFiles(selectedFiles);
            void handleUpload(selectedFiles);
            e.target.value = '';
          }}
        />
        {files.map((f) => (
          <span
            key={f.name}
            className="flex items-center gap-1 bg-primary-50 text-primary-700 px-2 py-1 rounded text-xs"
          >
            <FileText className="w-3 h-3" /> {f.name}
            <button onClick={() => setFiles(files.filter((x) => x !== f))}>
              <X className="w-3 h-3" />
            </button>
          </span>
        ))}
        {uploading && (
          <span className="ml-auto text-sm text-slate-500">上传中...</span>
        )}
      </div>
    </div>
  );
}
