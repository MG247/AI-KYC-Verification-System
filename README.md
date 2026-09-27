# KYC AI System

KYC AI System is a full-stack document review application for OCR extraction, document classification, field validation, fraud/tamper analysis, risk scoring, audit logging, and human review. The project is organized as a Python FastAPI backend plus a React/Vite frontend.

The application supports fixed KYC document schemas such as PAN, Aadhaar, Passport, Driving License, and GST Certificate. It also supports dynamic analysis for documents outside the fixed list, such as bank account information, invoices, utility bills, certificates, and other business proof documents. For those non-listed documents, the system returns a human-readable document label and a dynamic JSON extraction instead of forcing the document into a wrong fixed schema.

## Project Structure

```text
projectKYC/
	backend/
		app/                         FastAPI app, LangGraph workflow, services, schemas
		input/                       Local sample inputs, ignored except sample_user.json
		output/                      Runtime/generated OCR and report output, ignored by git
		tests/                       Backend unit tests
		.env                         Local secrets and runtime config, ignored by git
		.env.example                 Safe config template
		requirements.txt             Python dependencies
		run_e2e_test.py              Local graph smoke test over backend/input
		run_ocr_test.py              OCR-focused local smoke test
	frontend/
		src/
			api/                       Frontend API client
			components/                Shared UI/result components
			data/                      Static options and default sample payloads
			pages/                     One React file per page/workflow
			utils/                     Formatting and JSON shaping helpers
			App.tsx                    Thin app shell and navigation
		.env.example                 Safe frontend config template
		package.json                 Frontend scripts and dependencies
		vite.config.ts               Vite config and local API proxy
	.gitignore                     Secret/cache/build/runtime ignores
	.env.example                  Combined environment reference for GitHub
	README.md                      This documentation
```

## Main Features

- Full KYC verification through `POST /verify`
- OCR-only extraction through `POST /v1/ocr`
- AI-generated/tamper forensic detection through `POST /v1/detect-ai`
- Batch verification through `POST /batch-verify`
- Audit lookup through `GET /audit/{request_id}`
- In-process health and metrics endpoints
- Local PaddleOCR engine by default
- Optional GPT Vision OCR through Azure OpenAI
- Dynamic JSON extraction for non-listed document types
- React UI pages split by workflow: Verify, Tools, Batch, Audit

## Document Handling

### Fixed KYC Schemas

The backend has strict extractors and validators for:

- PAN
- Aadhaar
- Passport
- Driving License
- GST Certificate

These documents produce typed fields such as `pan_number`, `aadhaar_last4`, `passport_number`, `gstin`, `name`, and `dob`. The validator compares extracted fields with the provided `user_data` payload when matching ground-truth keys are available.

### Dynamic Documents

If the uploaded file is not one of the fixed KYC documents, the backend does not force a wrong schema. Instead it returns:

- `document_type`: usually `UNKNOWN` for unsupported fixed schemas
- `document_type_label`: readable label such as `Bank Account Information` or `Udyam Registration Certificate`
- `dynamic_extracted_data`: key-value JSON generated from the document
- `generic_document_analysis`: summary, category, confidence, extraction notes, and visual concerns

Fraud, AI-generated, metadata, and tamper checks still run for these dynamic documents.

## OCR Engines

Every OCR-capable endpoint accepts an `ocr_engine` value:

- `paddle`: local PaddleOCR engine, default
- `gpt_vision`: Azure OpenAI vision-based OCR and extraction

GPT Vision is useful when PaddleOCR misses low-quality text, layout-heavy documents, or non-standard business documents. Azure OpenAI configuration must be present in `backend/.env` for GPT Vision to work.

## Environment Variables

Do not put secrets in source files. Put local backend secrets only in `backend/.env` and local frontend overrides only in `frontend/.env.local`; both are ignored by git.

This repo includes three safe templates:

- `.env.example`: combined reference for GitHub reviewers and deployment setup
- `backend/.env.example`: copy to `backend/.env` for the FastAPI app
- `frontend/.env.example`: copy to `frontend/.env.local` for the Vite app

Backend local setup:

```bash
cd backend
copy .env.example .env
```

Frontend local setup:

```bash
cd frontend
copy .env.example .env.local
```

Local PaddleOCR works without Azure credentials. Fill Azure OpenAI values only when you want GPT Vision OCR, generic document analysis, or LLM review.

Important variables:

```env
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_API_VERSION=2024-10-21
KYC_LLM_ENABLE=true
KYC_LOG_LEVEL=INFO
KYC_AUDIT_FILE=audit.jsonl
KYC_OCR_PADDLE_DEVICE=cpu
VITE_API_BASE_URL=/api
```

OCR settings are configured through `KYC_OCR_*` variables in `backend/.env.example`.

## Backend Setup

From the repository root:

```bash
cd backend
python -m venv ..\.venv
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the API from the `backend/` folder:

```bash
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
Invoke-RestMethod http://localhost:8000/healthz
```

## Frontend Setup

From the repository root:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173/
```

The Vite dev server proxies `/api/*` to `http://localhost:8000`.

The default frontend env uses the Vite `/api` proxy. For a hosted backend, update `frontend/.env.local`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

## Frontend Pages

Each workflow now lives in a separate React file under `frontend/src/pages/`:

- `VerifyPage.tsx`: full pipeline verification by upload or server path
- `ToolsPage.tsx`: OCR-only and AI forensic tools
- `BatchPage.tsx`: batch verification JSON runner
- `AuditPage.tsx`: audit lookup by request ID

Reusable UI and result rendering live in `frontend/src/components/`.

## API Reference

### `GET /healthz`

Returns API, OCR, and LLM readiness.

### `GET /metrics`

Returns in-process counts and latency samples.

### `POST /v1/ocr`

Multipart OCR-only extraction.

Fields:

- `file`: document image/PDF
- `document_type_hint`: optional fixed schema hint
- `ocr_engine`: `paddle` or `gpt_vision`

### `POST /v1/detect-ai`

Runs AI-generated/tamper forensic detection.

Fields:

- `file`: optional upload
- `document_path`: optional server-local file path

### `POST /verify`

Runs the full LangGraph pipeline.

Multipart upload fields:

- `file`
- `user_data`: JSON string
- `document_type_hint`
- `ocr_engine`

JSON body example:

```json
{
	"document_path": "input/pan.jpg.jpeg",
	"user_data": {
		"name": "Mohit Gautam",
		"dob": "XXXX-XX-XX",
		"pan": "XXXXXXXXXXX"
	},
	"document_type_hint": "PAN",
	"ocr_engine": "paddle"
}
```

### `POST /batch-verify`

Runs up to 20 server-local documents.

```json
{
	"ocr_engine": "paddle",
	"user_data": {},
	"requests": [
		{"document_path": "input/pan.jpg.jpeg", "document_type_hint": "PAN"},
		{"document_path": "input/aadhar.jpg.jpeg", "document_type_hint": "AADHAAR"}
	]
}
```

### `GET /audit/{request_id}`

Returns the stored audit record for a previous decision.

## Validation And Tests

Backend checks:

```bash
cd backend
..\.venv\Scripts\python.exe -m compileall app
..\.venv\Scripts\python.exe -m pytest tests
```

Frontend checks:

```bash
cd frontend
npm run lint
npm run build
```

## Security And Git Hygiene

The root `.gitignore` excludes:

- `.env` files and local secrets, except safe `.env.example` templates
- Python virtual environments and caches
- frontend `node_modules/` and `dist/`
- backend runtime output and audit logs
- local input documents that may contain customer PII
- temporary files, logs, and OS/editor noise

Never commit `backend/.env`, customer documents, generated audit logs, or runtime output.

## Removed Integrations

The previous Pine Labs opportunity API runner and downloaded opportunity artifacts have been removed. The project now focuses on local/API document verification through the generic KYC pipeline and frontend console.
