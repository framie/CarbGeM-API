﻿# Zone of Inhibition Measurement API

This project provides an Express.js server (written in TypeScript) that accepts image uploads via an API, stores them, and invokes a Python script to process antibiotic disk inhibition zones on an agar plate.

## Prerequisites

- **Node.js** v18+ with `npm`
- **Python 3** (recommended: ≥ 3.8)
- **Python packages**:
  - `opencv-python`
  - `numpy`

Install Python packages using pip:
```bash
pip install opencv-python numpy
```

## Setup & Run

### 1. Install Node Packages
```bash
npm install
```

### 2. Create `.env`

```env
PORT=3000
PYTHON_PATH=C:/Path/To/python.exe
```

### 3. Start the server (dev mode)
```bash
npm run dev
```


## API: Analyze Image

### `POST /analyze`

- Accepts: multipart/form-data with field name `image`
- Behavior:
  - Stores image (if not already stored)
  - Runs Python script to analyze image
  - Returns output image

#### Example with `curl`:

```bash
curl -F 'image=@sample.png' http://localhost:3000/analyze
```
