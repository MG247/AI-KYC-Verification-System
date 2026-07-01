import type { OcrEngine } from '../types'

export const documentHints = [
  'PAN',
  'AADHAAR',
  'PASSPORT',
  'DRIVING_LICENSE',
  'GST_CERTIFICATE',
]

export const ocrEngines: Array<{ value: OcrEngine; label: string }> = [
  { value: 'paddle', label: 'PaddleOCR' },
  { value: 'gpt_vision', label: 'GPT Vision' },
]

export const sampleProfile = {
  name: 'Mohit Gautam',
  dob: '2006-08-10',
  pan: 'ERHPG7459D',
  aadhaar: '2457 7114 0321',
  document_type: 'PAN',
}

export const defaultBatchPayload = {
  ocr_engine: 'paddle',
  user_data: sampleProfile,
  requests: [
    {
      document_path: 'input/pan.jpg.jpeg',
      document_type_hint: 'PAN',
    },
    {
      document_path: 'input/aadhar.jpg.jpeg',
      document_type_hint: 'AADHAAR',
    },
  ],
}
