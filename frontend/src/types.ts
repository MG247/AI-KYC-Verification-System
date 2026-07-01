export type TabKey = 'verify' | 'tools' | 'batch' | 'audit'
export type OcrEngine = 'paddle' | 'gpt_vision'

export type HealthResponse = {
  status: string
  ocr: Record<string, unknown>
  llm: Record<string, unknown>
  version: string
}

export type MetricsResponse = {
  requests_total: number
  errors_total: number
  decisions: Record<string, number>
  latency_ms: {
    p50: number
    p95: number
    samples: number
  }
}

export type KycResponse = {
  request_id: string
  document_type: string
  document_type_label?: string | null
  ocr_engine: string
  classification_confidence: number
  final_decision: string
  risk_band: string
  risk_score: number
  validation_overall: string
  field_match_score: number
  fraud_verdict: string
  fraud_score: number
  ocr_confidence: number
  extracted_data: Record<string, unknown>
  typed_extracted_data: Record<string, unknown>
  corrected_fields: Record<string, unknown>
  dynamic_extracted_data: Record<string, unknown>
  generic_document_analysis: Record<string, unknown>
  document_data: Record<string, unknown>
  validation_results: Record<string, unknown>
  fraud_reasons: string[]
  metadata_results: Record<string, unknown>
  preprocess_report: Record<string, unknown>
  decision_reasons: string[]
  errors: string[]
  timings: Record<string, number>
  processing_time: number
  audit_logs: Array<Record<string, unknown>>
  hallucination_score: number
  document_status: string
  document_feedback: Array<Record<string, unknown>>
  llm_enabled: boolean
  llm_recommendation?: string | null
  llm_confidence?: number | null
  llm_reasoning?: string | null
  llm_concerns: string[]
  llm_rectifications: Array<Record<string, unknown>>
}

export type BatchResponse = {
  count: number
  results: Array<KycResponse | Record<string, unknown>>
}
