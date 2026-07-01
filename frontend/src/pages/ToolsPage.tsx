import { Bot, FileScan, ImageUp, Loader2 } from 'lucide-react'
import { useState } from 'react'
import type { FormEvent } from 'react'
import { requestJson } from '../api/client'
import { InlineError, JsonBlock, PanelHeader } from '../components/common'
import { OcrOutputPanel } from '../components/OcrOutputPanel'
import { documentHints, ocrEngines } from '../data/defaults'
import type { OcrEngine } from '../types'
import { formatLabel } from '../utils/format'

export function ToolsPage() {
  const [ocrFile, setOcrFile] = useState<File | null>(null)
  const [ocrHint, setOcrHint] = useState('')
  const [toolOcrEngine, setToolOcrEngine] = useState<OcrEngine>('paddle')
  const [ocrResult, setOcrResult] = useState<unknown>(null)
  const [ocrError, setOcrError] = useState<string | null>(null)
  const [isOcrRunning, setIsOcrRunning] = useState(false)

  const [aiFile, setAiFile] = useState<File | null>(null)
  const [aiPath, setAiPath] = useState('')
  const [aiResult, setAiResult] = useState<unknown>(null)
  const [aiError, setAiError] = useState<string | null>(null)
  const [isAiRunning, setIsAiRunning] = useState(false)

  async function handleOcrSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setOcrError(null)
    setIsOcrRunning(true)

    try {
      if (!ocrFile) {
        throw new Error('Choose a document file.')
      }
      const body = new FormData()
      body.append('file', ocrFile)
      body.append('ocr_engine', toolOcrEngine)
      if (ocrHint) {
        body.append('document_type_hint', ocrHint)
      }
      setOcrResult(await requestJson<unknown>('/v1/ocr', { method: 'POST', body }))
    } catch (error) {
      setOcrError(error instanceof Error ? error.message : 'OCR failed.')
    } finally {
      setIsOcrRunning(false)
    }
  }

  async function handleAiSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setAiError(null)
    setIsAiRunning(true)

    try {
      const body = new FormData()
      if (aiFile) {
        body.append('file', aiFile)
      } else if (aiPath.trim()) {
        body.append('document_path', aiPath.trim())
      } else {
        throw new Error('Choose a file or enter a server document path.')
      }
      setAiResult(await requestJson<unknown>('/v1/detect-ai', { method: 'POST', body }))
    } catch (error) {
      setAiError(error instanceof Error ? error.message : 'AI detection failed.')
    } finally {
      setIsAiRunning(false)
    }
  }

  return (
    <section className="content-grid split-tools">
      <form className="panel form-panel" onSubmit={handleOcrSubmit}>
        <PanelHeader icon={<ImageUp size={20} />} title="OCR Extraction" eyebrow="POST /v1/ocr" />
        <label className="field file-field">
          <span>Document File</span>
          <input type="file" accept="image/*,.pdf" onChange={(event) => setOcrFile(event.target.files?.[0] ?? null)} />
        </label>
        <label className="field">
          <span>Document Hint</span>
          <select value={ocrHint} onChange={(event) => setOcrHint(event.target.value)}>
            <option value="">Auto detect</option>
            {documentHints.map((hint) => (
              <option key={hint} value={hint}>
                {formatLabel(hint)}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>OCR Engine</span>
          <select value={toolOcrEngine} onChange={(event) => setToolOcrEngine(event.target.value as OcrEngine)}>
            {ocrEngines.map((engine) => (
              <option key={engine.value} value={engine.value}>
                {engine.label}
              </option>
            ))}
          </select>
        </label>
        {ocrError && <InlineError message={ocrError} />}
        <button className="primary-button secondary" type="submit" disabled={isOcrRunning}>
          {isOcrRunning ? <Loader2 className="spin" size={17} /> : <FileScan size={17} />}
          Extract OCR
        </button>
        <OcrOutputPanel result={ocrResult} />
      </form>

      <form className="panel form-panel" onSubmit={handleAiSubmit}>
        <PanelHeader icon={<Bot size={20} />} title="AI Document Check" eyebrow="POST /v1/detect-ai" />
        <label className="field file-field">
          <span>Document File</span>
          <input type="file" accept="image/*,.pdf" onChange={(event) => setAiFile(event.target.files?.[0] ?? null)} />
        </label>
        <label className="field">
          <span>Server Path</span>
          <input value={aiPath} onChange={(event) => setAiPath(event.target.value)} />
        </label>
        {aiError && <InlineError message={aiError} />}
        <button className="primary-button secondary" type="submit" disabled={isAiRunning}>
          {isAiRunning ? <Loader2 className="spin" size={17} /> : <Bot size={17} />}
          Check Document
        </button>
        <JsonBlock data={aiResult} emptyText="AI detection result will appear here." />
      </form>
    </section>
  )
}
