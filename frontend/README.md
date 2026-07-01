# KYC AI Console Frontend

React + Vite interface for the FastAPI KYC pipeline.

## Run

Start the API from the repository root:

```bash
uvicorn app.main:app --reload --port 8000
```

Start the frontend from this folder:

```bash
npm run dev
```

The Vite server proxies `/api/*` to `http://localhost:8000`. Set `VITE_API_BASE_URL` when the API is hosted elsewhere.

## OCR Engines

The UI can send `ocr_engine=paddle` or `ocr_engine=gpt_vision` to `/verify` and `/v1/ocr`.

- `paddle`: local PaddleOCR engine, used by default.
- `gpt_vision`: Azure OpenAI vision OCR, useful for documents where PaddleOCR misses layout or low-quality text.

For non-listed documents, the API returns `document_type_label`, `dynamic_extracted_data`, and `generic_document_analysis` so reviewers can inspect a dynamic JSON extraction while fraud and metadata checks still run.
