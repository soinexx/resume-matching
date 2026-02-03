# TEAM7 Resume Analysis Platform

AI-powered resume analysis system with intelligent job matching, multi-language support, and analytics dashboard.

## Features

- **Resume Upload & Analysis**: Support for PDF and DOCX formats
- **Unified Matching System**: Three-method matching combining Keyword, TF-IDF, and Vector similarity
- **AI-Powered Insights**: Semantic understanding with sentence-transformers
- **Multi-language**: English and Russian support
- **Analytics Dashboard**: Hiring funnels, skill demand, recruiter performance, model quality metrics
- **Async Processing**: Celery + Redis for background tasks
- **Modern UI**: React 18 + Material-UI with responsive design

## Quick Start

Choose your operating system:

### macOS / Linux

```bash
git clone https://github.com/soinexx/resume-matching.git
cd resume-matching
bash setup.sh
```

### Windows (PowerShell)

```powershell
git clone https://github.com/soinexx/resume-matching.git
cd resume-matching
.\setup.ps1
```

Then open http://localhost:5173

### Load Test Data (Optional)

```bash
# macOS/Linux
bash scripts/load_test_data.sh

# Windows
.\scripts\load-test-data.ps1
```

This uploads 65 sample resumes and 5 job vacancies.

## Requirements

- **Docker Desktop** (Mac/Windows) or Docker + Docker Compose (Linux)
- **8GB RAM** minimum (16GB recommended)
- **5GB disk space**

## Services

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:5173 | React UI |
| Backend API | http://localhost:8000 | FastAPI backend |
| API Docs | http://localhost:8000/docs | Interactive documentation |
| Analytics | http://localhost:5173/recruiter/analytics | Hiring metrics dashboard |
| Flower | http://localhost:5555 | Celery monitoring |

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  Frontend   │─────▶│   Backend    │─────▶│   Database  │
│ (React+MUI) │      │   (FastAPI)  │      │ (PostgreSQL)│
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                    ┌───────┴────────┐
                    ▼                ▼
              ┌─────────┐      ┌──────────┐
              │ Celery  │      │  Redis   │
              │ Worker  │      │  Broker  │
              └─────────┘      └──────────┘
```

## How It Works

### 1. Resume Upload & Parsing

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Upload PDF/  │────▶│ Extract Text │────▶│ Save to DB   │
│   DOCX       │     │ (PyPDF2/     │     │ (status:     │
│              │     │  python-docx)│     │  uploaded)   │
└──────────────┘     └──────────────┘     └──────────────┘
```

### 2. Resume Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     RESUME ANALYSIS PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. LANGUAGE DETECTION (langdetect)                            │
│     └─▶ Detects English or Russian                             │
│                                                                 │
│  2. KEYWORD EXTRACTION (KeyBERT)                                │
│     └─▶ Extracts key skills and competencies                   │
│                                                                 │
│  3. NAMED ENTITY RECOGNITION (SpaCy)                            │
│     ├─▶ en_core_web_sm (English)                               │
│     └─▶ ru_core_news_sm (Russian)                              │
│         • Organizations (companies)                            │
│         • Dates (work periods)                                  │
│         • Technical skills                                      │
│         • Person names                                          │
│                                                                 │
│  4. EXPERIENCE CALCULATION                                     │
│     ├─▶ Parse work periods from dates                          │
│     ├─▶ Detect overlapping periods (avoid double-count)       │
│     └─▶ Calculate total years/months of experience             │
│                                                                 │
│  5. GRAMMAR CHECKING (LanguageTool)                            │
│     ├─▶ Grammar errors                                          │
│     ├─▶ Spelling mistakes                                      │
│     └─▶ Style suggestions                                       │
│                                                                 │
│  6. ERROR DETECTION                                            │
│     ├─▶ Missing contact info                                    │
│     ├─▶ Resume too short                                        │
│     ├─▶ No portfolio (for juniors)                             │
│     └─▶ Inconsistent dates                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Unified Job Matching Algorithm

```
┌─────────────────────────────────────────────────────────────────┐
│                 UNIFIED MATCHING SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  VACANCY REQUIREMENTS + RESUME                                   │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              1. KEYWORD MATCHING (50%)                   │   │
│  │  • Direct match (python → Python)                       │   │
│  │  • Synonym match (unix → Linux, bash)                   │   │
│  │  • Fuzzy matching (ReactJS → React)                      │   │
│  │  • Compound skills (machine learning → ML)               │   │
│  └────────────────────┬────────────────────────────────────┘   │
│                       │                                          │
│  ┌────────────────────┴────────────────────────────────────┐   │
│  │              2. TF-IDF MATCHING (30%)                    │   │
│  │  • Term Frequency-Inverse Document Frequency            │   │
│  │  • Ranks skills by importance in vacancy                │   │
│  │  • Weighted scoring based on keyword relevance          │   │
│  └────────────────────┬────────────────────────────────────┘   │
│                       │                                          │
│  ┌────────────────────┴────────────────────────────────────┐   │
│  │              3. VECTOR SEMANTIC (20%)                   │   │
│  │  • sentence-transformers (all-MiniLM-L6-v2)            │   │
│  │  • Semantic similarity of resume vs vacancy            │   │
│  │  • Cosine similarity: -1 to 1 (normalized to 0-1)      │   │
│  └────────────────────┬────────────────────────────────────┘   │
│                       │                                          │
│                       ▼                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              OVERALL SCORE + RECOMMENDATION              │   │
│  │  • 0-100% match percentage                               │   │
│  │  • excellent (≥80%), good (≥60%), maybe (≥40%), poor   │   │
│  │  • Matched/missing skills with detailed breakdown      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Skill Synonym Matching

The system understands that different terms can mean the same skill:

| User Input | Matches Also |
|-----------|--------------|
| PostgreSQL | SQL, Postgres, psql |
| React | ReactJS, React.js, React.js |
| JavaScript | JS, javascript |
| Java | Java 8, Java 11, Java 17 |

## Common Commands

```bash
# View logs
docker compose logs -f

# View specific service logs
docker compose logs backend
docker compose logs frontend

# Restart services
docker compose restart

# Stop all services
docker compose down

# Stop and remove data
docker compose down -v
```

## Tech Stack

### Backend
- **Framework**: FastAPI with Python 3.11+
- **Database**: PostgreSQL 14 with SQLAlchemy 2.0
- **ML/NLP**: KeyBERT, SpaCy, LanguageTool, sentence-transformers
- **Matching**: TF-IDF (scikit-learn), Vector similarity (all-MiniLM-L6-v2)
- **Async**: Celery + Redis

### Frontend
- **Framework**: React 18 with TypeScript 5.6
- **Build**: Vite 5.4
- **UI**: Material-UI (MUI) v6
- **i18n**: react-i18next (EN/RU)

## Project Structure

```
├── backend/               # FastAPI backend
│   ├── analyzers/         # ML/NLP analyzers
│   │   ├── enhanced_matcher.py    # Keyword + synonym matching
│   │   ├── tfidf_matcher.py       # TF-IDF weighted matching
│   │   ├── vector_matcher.py      # Semantic similarity
│   │   ├── unified_matcher.py     # Combined 3-method matcher
│   │   └── hf_skill_extractor.py  # HuggingFace skill extraction
│   ├── api/               # API endpoints
│   ├── tasks/             # Celery tasks
│   ├── models/            # SQLAlchemy models
│   └── alembic/           # Database migrations
├── frontend/              # React + Vite frontend
│   ├── src/
│   │   ├── api/           # API client
│   │   ├── components/    # React components
│   │   │   └── UnifiedMatchMetrics.tsx  # Unified metrics display
│   │   ├── pages/         # Page components
│   │   └── i18n/          # Translations (EN/RU)
│   └── nginx.conf         # nginx config for production
├── scripts/               # Setup and utility scripts
│   ├── reset_and_reload.py    # Database reset script
│   └── load_test_data.sh       # Test data loader
├── services/              # Shared services
│   └── data_extractor/    # PDF/DOCX extraction
├── docker-compose.yml     # Docker services
├── setup.sh               # Setup script (Mac/Linux)
├── setup.ps1              # Setup script (Windows)
├── .env.example           # Environment template
├── README.md              # This file
└── SETUP.md               # Detailed setup guide
```

## API Examples

### Upload Resume

```bash
curl -X POST http://localhost:8000/api/resumes/upload \
  -F "file=@resume.pdf"
```

### Analyze Resume

```bash
curl -X POST http://localhost:8000/api/resumes/analyze \
  -H "Content-Type: application/json" \
  -d '{"resume_id": "uuid", "extract_keywords": true}'
```

### Job Matching (Simple)

```bash
curl -X POST http://localhost:8000/api/matching/compare \
  -H "Content-Type: application/json" \
  -d '{
    "resume_id": "uuid",
    "vacancy_data": {
      "position": "Java Developer",
      "mandatory_requirements": ["Java", "Spring"]
    }
  }'
```

### Job Matching (Unified - AI-Powered)

```bash
curl -X POST http://localhost:8000/api/matching/compare-unified \
  -H "Content-Type: application/json" \
  -d '{
    "resume_id": "uuid",
    "vacancy_data": {
      "id": "vacancy_uuid",
      "title": "Python Developer",
      "description": "We are looking for...",
      "required_skills": ["python", "django", "postgresql"]
    }
  }'
```

**Response:**
```json
{
  "overall_score": 0.75,
  "recommendation": "good",
  "keyword_score": 0.80,
  "tfidf_score": 0.65,
  "vector_score": 0.72,
  "vector_similarity": 0.51,
  "matched_skills": ["python", "django"],
  "missing_skills": ["postgresql"],
  "tfidf_matched": ["python", "developer", "api"],
  "tfidf_missing": ["postgresql", "database"]
}
```

## Documentation

- [Setup Guide](SETUP.md) - Detailed installation instructions
- [Russian README](README_RU.md) - Версия на русском языке
- [ML Pipeline Details](ML_PIPELINE.md) - How the ML/NLP analysis works
- [Database Setup](backend/DATABASE_SETUP.md) - Database configuration
- [Matching System Guide](backend/analyzers/MATCHERS_GUIDE.md) - Unified matching system details
- [HF Extractor README](backend/analyzers/HF_EXTRACTOR_README.md) - Skill extraction documentation

## Troubleshooting

**Port already in use?**
Edit `.env` and change `FRONTEND_PORT` or `BACKEND_PORT`.

**Services not starting?**
```bash
docker compose logs backend
```

**PowerShell scripts blocked?**
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Reset everything?**
```bash
docker compose down -v
bash setup.sh
```

## License

MIT

---

Built by TEAM7
