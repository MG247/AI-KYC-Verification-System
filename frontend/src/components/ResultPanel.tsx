import { ShieldCheck } from 'lucide-react'
import type { KycResponse } from '../types'
import { decisionTone, formatLabel, isRecord } from '../utils/format'
import { EmptyState, FeedbackList, JsonBlock, KeyValueGrid, ReasonList, ScoreCard, StatusBadge } from './common'

export function ResultPanel({ result }: { result: KycResponse | null }) {
  if (!result) {
    return (
      <section className="panel results-panel empty-results">
        <EmptyState icon={<ShieldCheck size={28} />} text="Verification result will appear here." />
      </section>
    )
  }

  const documentLabel = result.document_type_label || result.document_type
  const genericCategory = isRecord(result.generic_document_analysis)
    ? String(result.generic_document_analysis.document_category ?? 'OTHER')
    : 'OTHER'

  return (
    <section className="panel results-panel">
      <div className="result-hero">
        <div>
          <span className="eyebrow">{result.request_id}</span>
          <h2>{formatLabel(result.final_decision)}</h2>
          <p>{formatLabel(result.document_status)} · {formatLabel(documentLabel)}</p>
        </div>
        <StatusBadge value={result.final_decision} />
      </div>

      <div className="score-grid">
        <ScoreCard label="Risk" value={result.risk_score} detail={result.risk_band} tone={decisionTone(result.risk_band)} />
        <ScoreCard label="OCR" value={result.ocr_confidence} detail="Confidence" tone="success" />
        <ScoreCard label="Fields" value={result.field_match_score} detail={result.validation_overall} tone={decisionTone(result.validation_overall)} />
        <ScoreCard label="Fraud" value={result.fraud_score} detail={result.fraud_verdict} tone={decisionTone(result.fraud_verdict)} invert />
      </div>

      <section className="result-section">
        <h3>Document Review</h3>
        <KeyValueGrid
          data={{
            detected_type: result.document_type,
            detected_label: documentLabel,
            category: genericCategory,
            ocr_engine: result.ocr_engine,
            classification_confidence: result.classification_confidence,
          }}
        />
      </section>

      <section className="result-section">
        <h3>Document Data</h3>
        <KeyValueGrid data={result.document_data} />
      </section>

      {Object.keys(result.dynamic_extracted_data || {}).length > 0 && (
        <section className="result-section">
          <h3>Dynamic JSON</h3>
          <JsonBlock data={result.dynamic_extracted_data} />
        </section>
      )}

      <section className="result-section">
        <h3>Decision Reasons</h3>
        <ReasonList items={result.decision_reasons} fallback="No decision reasons returned." />
      </section>

      <section className="result-section">
        <h3>Fraud Reasons</h3>
        <ReasonList items={result.fraud_reasons} fallback="No fraud reasons returned." />
      </section>

      <section className="result-section">
        <h3>Document Feedback</h3>
        <FeedbackList items={result.document_feedback} />
      </section>

      <details className="details-block">
        <summary>Validation Results</summary>
        <JsonBlock data={result.validation_results} />
      </details>

      <details className="details-block">
        <summary>Metadata, Timings, and LLM</summary>
        <JsonBlock
          data={{
            document_type: result.document_type,
            document_type_label: result.document_type_label,
            ocr_engine: result.ocr_engine,
            classification_confidence: result.classification_confidence,
            generic_document_analysis: result.generic_document_analysis,
            metadata_results: result.metadata_results,
            preprocess_report: result.preprocess_report,
            timings: result.timings,
            processing_time: result.processing_time,
            hallucination_score: result.hallucination_score,
            llm: {
              enabled: result.llm_enabled,
              recommendation: result.llm_recommendation,
              confidence: result.llm_confidence,
              reasoning: result.llm_reasoning,
              concerns: result.llm_concerns,
              rectifications: result.llm_rectifications,
            },
            errors: result.errors,
          }}
        />
      </details>
    </section>
  )
}
