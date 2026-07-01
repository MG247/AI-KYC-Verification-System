import { Braces, FileScan, Loader2, ShieldCheck, UploadCloud } from 'lucide-react'
import { useState } from 'react'
import type { FormEvent } from 'react'
import { requestJson } from '../api/client'
import { PanelHeader, InlineError } from '../components/common'
import { ResultPanel } from '../components/ResultPanel'
import { documentHints, ocrEngines, sampleProfile } from '../data/defaults'
import type { KycResponse, OcrEngine } from '../types'
import { formatLabel, parseJsonObject } from '../utils/format'

export function VerifyPage({
  onVerified,
  onStatusRefresh,
}: {
  onVerified: (result: KycResponse) => void
  onStatusRefresh: () => Promise<void>
}) {
  const [verifyMode, setVerifyMode] = useState<'upload' | 'serverPath'>('upload')
  const [verifyFile, setVerifyFile] = useState<File | null>(null)
  const [serverPath, setServerPath] = useState('input/pan.jpg.jpeg')
  const [requestId, setRequestId] = useState('')
  const [documentHint, setDocumentHint] = useState('PAN')
  const [verifyOcrEngine, setVerifyOcrEngine] = useState<OcrEngine>('paddle')
  const [profileJson, setProfileJson] = useState(JSON.stringify(sampleProfile, null, 2))
  const [verifyError, setVerifyError] = useState<string | null>(null)
  const [kycResult, setKycResult] = useState<KycResponse | null>(null)
  const [isVerifying, setIsVerifying] = useState(false)

  async function handleVerifySubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setVerifyError(null)
    setIsVerifying(true)

    try {
      const userData = parseJsonObject(profileJson, 'Profile data')
      let result: KycResponse

      if (verifyMode === 'upload') {
        if (!verifyFile) {
          throw new Error('Choose a document file.')
        }
        const body = new FormData()
        body.append('file', verifyFile)
        body.append('user_data', JSON.stringify(userData))
        body.append('ocr_engine', verifyOcrEngine)
        if (documentHint) {
          body.append('document_type_hint', documentHint)
        }
        result = await requestJson<KycResponse>('/verify', { method: 'POST', body })
      } else {
        if (!serverPath.trim()) {
          throw new Error('Enter a server document path.')
        }
        result = await requestJson<KycResponse>('/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            document_path: serverPath.trim(),
            user_data: userData,
            document_type_hint: documentHint || undefined,
            ocr_engine: verifyOcrEngine,
            request_id: requestId.trim() || undefined,
          }),
        })
      }

      setKycResult(result)
      onVerified(result)
      await onStatusRefresh()
    } catch (error) {
      setVerifyError(error instanceof Error ? error.message : 'Verification failed.')
    } finally {
      setIsVerifying(false)
    }
  }

  return (
    <section className="content-grid">
      <form className="panel form-panel" onSubmit={handleVerifySubmit}>
        <PanelHeader icon={<FileScan size={20} />} title="Full KYC Verification" eyebrow="POST /verify" />

        <div className="segmented-control" aria-label="Verification source">
          <button
            type="button"
            className={verifyMode === 'upload' ? 'selected' : ''}
            onClick={() => setVerifyMode('upload')}
          >
            <UploadCloud size={16} />
            Upload
          </button>
          <button
            type="button"
            className={verifyMode === 'serverPath' ? 'selected' : ''}
            onClick={() => setVerifyMode('serverPath')}
          >
            <Braces size={16} />
            Server Path
          </button>
        </div>

        {verifyMode === 'upload' ? (
          <label className="field file-field">
            <span>Document File</span>
            <input
              type="file"
              accept="image/*,.pdf"
              onChange={(event) => setVerifyFile(event.target.files?.[0] ?? null)}
            />
          </label>
        ) : (
          <div className="two-column">
            <label className="field">
              <span>Document Path</span>
              <input value={serverPath} onChange={(event) => setServerPath(event.target.value)} />
            </label>
            <label className="field">
              <span>Request ID</span>
              <input value={requestId} onChange={(event) => setRequestId(event.target.value)} />
            </label>
          </div>
        )}

        <div className="two-column">
          <label className="field">
            <span>Document Hint</span>
            <select value={documentHint} onChange={(event) => setDocumentHint(event.target.value)}>
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
            <select value={verifyOcrEngine} onChange={(event) => setVerifyOcrEngine(event.target.value as OcrEngine)}>
              {ocrEngines.map((engine) => (
                <option key={engine.value} value={engine.value}>
                  {engine.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <label className="field">
          <span>Profile JSON</span>
          <textarea value={profileJson} onChange={(event) => setProfileJson(event.target.value)} rows={9} />
        </label>

        {verifyError && <InlineError message={verifyError} />}

        <button className="primary-button" type="submit" disabled={isVerifying}>
          {isVerifying ? <Loader2 className="spin" size={17} /> : <ShieldCheck size={17} />}
          Run Verification
        </button>
      </form>

      <ResultPanel result={kycResult} />
    </section>
  )
}
