# SimilarPic Backend

FastAPI backend for fashion image classification and recommendation system.

## Features

- **M1 (Category Classification)**: Classify fashion items into categories (T-Shirt, Jeans, Dress, etc.)
- **M2 (Recommendation)**: Find similar fashion items using CLIP embeddings and FAISS
- **M3 (Style Classification)**: Classify fashion items by style (Casual, Formal, Ethnic, etc.)

## Project Structure

```
similarpic-server/
├── app/
│   ├── main.py                    # FastAPI application
│   ├── api/                       # API routes
│   │   ├── deps.py                # Dependency injection
│   │   └── v1/                    # API v1 endpoints
│   │       ├── routes_category.py
│   │       ├── routes_style.py
│   │       ├── routes_recommend.py
│   │       └── routes_health.py
│   ├── core/                      # Core configuration
│   │   ├── config.py              # Settings
│   │   └── lifespan.py            # Model loading
│   ├── ml/                        # ML models
│   │   ├── transforms.py
│   │   ├── classifier.py
│   │   ├── clip_encoder.py
│   │   └── recommender.py
│   ├── schemas/                   # Pydantic schemas
│   │   ├── common.py
│   │   ├── classify.py
│   │   └── recommend.py
│   └── services/                  # Business logic
│       ├── image_service.py
│       └── metadata_service.py
├── models/                        # ML model weights (not in git)
├── data/                          # CSV data files (not in git)
├── tests/                         # Test files
├── requirements.txt
├── pyproject.toml                 # Project configuration
├── Dockerfile
├── docker-compose.yml
└── download_clip_processor.py     # Script to download CLIP processor
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd similarpic-server
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional):
```bash
# Create .env file with your configuration
# See Configuration section below for available variables
```

5. Ensure models and data are in place:
- `models/clip/clip_model.pth` and `models/clip/clip_processor/`
- `models/classifiers/classifier_category_best.pth`
- `models/classifiers/classifier_style_best.pth`
- `models/faiss/faiss_index_flat.index`
- `models/faiss/img_id_mapping.json`
- `data/images.csv`
- `data/styles.csv`

## Running the Server

### Development

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Root
```bash
GET /
```

Returns API information and version.

### Health Check
```bash
GET /api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

### Classify Category
```bash
POST /api/v1/classify/category
Content-Type: multipart/form-data

file: <image file>
```

Response:
```json
{
  "prediction": "Topwear",
  "confidence": 0.94,
  "all_predictions": [
    {"label": "Topwear", "confidence": 0.94},
    {"label": "Bottomwear", "confidence": 0.03}
  ]
}
```

### Classify Style
```bash
POST /api/v1/classify/style
Content-Type: multipart/form-data

file: <image file>
```

Response:
```json
{
  "prediction": "Casual",
  "confidence": 0.87,
  "all_predictions": [
    {"label": "Casual", "confidence": 0.87},
    {"label": "Formal", "confidence": 0.10}
  ]
}
```

### Recommend Similar Items
```bash
POST /api/v1/recommend/similar?top_k=10
Content-Type: multipart/form-data

file: <image file>
```

Response:
```json
{
  "query_processed": true,
  "similar_items": [
    {
      "img_id": "15970",
      "score": 0.95,
      "image_url": "http://...",
      "product_name": "Turtle Check Men Navy Blue Shirt",
      "category": "Topwear",
      "color": "Navy Blue"
    }
  ],
  "total_results": 10
}
```

## Configuration

Environment variables (create a `.env` file in the project root):

### Server Configuration
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `DEBUG`: Debug mode (default: `False`)

### Paths
- `MODEL_DIR`: Path to models directory (default: `models`)
- `DATA_DIR`: Path to data directory (default: `data`)

### ML Configuration
- `DEVICE`: Inference device - `cpu` or `cuda` (default: `cpu`)
- `TOP_K_RECOMMENDATIONS`: Default number of recommendations (default: `10`, range: 1-50)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for predictions (default: `0.1`, range: 0.0-1.0)

### File Upload
- `MAX_FILE_SIZE`: Maximum upload size in bytes (default: `10485760` = 10MB)
- `ALLOWED_EXTENSIONS`: Comma-separated list of allowed extensions (default: `jpg,jpeg,png,webp`)

### CORS
- `CORS_ORIGINS`: Comma-separated list of allowed origins (default: `http://localhost:3000,http://localhost:5173,http://localhost:8000`)

## Docker

### Using Docker Compose (Recommended)

```bash
docker-compose up --build
```

This will:
- Build the Docker image
- Mount `models/` and `data/` directories as volumes
- Expose the API on port 8000
- Set up environment variables

### Using Docker directly

Build and run with Docker:

```bash
docker build -t similarpic-server .
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -e HOST=0.0.0.0 \
  -e PORT=8000 \
  -e DEVICE=cpu \
  similarpic-server
```

## Testing

Run tests:
```bash
pytest tests/
```

## Pre-trained Models

This backend uses the SimilarPic models trained by **aleec02**.
The models were obtained from the original repository:

👉 https://github.com/aleec02/TF-similarPic[https://github.com/aleec02/TF-similarPic]

The following artifacts are included in this repository to facilitate backend execution:

- Preprocessed CLIP model (clip_model.pth)

- Category classifier (classifier_category_best.pth)

- Style classifier (classifier_style_best.pth)

- FAISS index for recommendations (embeddings_index.faiss)

- Metadata files (img_id_mapping.json, embeddings_metadata.json, …)

## License

Unlicense license.
