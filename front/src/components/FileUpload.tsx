import { useRef, useState } from 'react';
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

  async function handleUpload() {
    if (files.length === 0) return;
    setUploading(true);
    try {
      await uploadFiles(sessionId, files);
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
    <div className="bg-white border-b border-slate-200 px-6 py-3">
      <div className="flex items-center gap-2 flex-wrap">
        <button
          onClick={() => inputRef.current?.click()}
          className="flex items-center gap-1 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-md text-sm"
        >
          <Upload className="w-4 h-4" /> 请选择文件
        </button>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".txt,.md,.csv"
          hidden
          onChange={(e) => {
            const selectedFiles = Array.from(e.target.files || []);
            const invalidFile = selectedFiles.find(
              (file) => file.size > MAX_FILE_SIZE,
            );
            if (invalidFile) {
              alert(`文件 ${invalidFile.name} 超过 10 MB 限制`);
              e.target.value = '';
              return;
            }
            setFiles(selectedFiles);
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
        {files.length > 0 && (
          <button
            onClick={handleUpload}
            disabled={uploading}
            className="ml-auto px-4 py-1.5 bg-primary-600 hover:bg-primary-700 text-white rounded-md text-sm disabled:opacity-50"
          >
            {uploading ? '上传中...' : `上传 ${files.length} 个文件`}
          </button>
        )}
      </div>
    </div>
  );
}
