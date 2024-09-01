# Documentation

## Pulling Repo

1. Install git lfs for large files

```bash
brew install git-lfs
```

2. Add lfs to git

```bash
git lfs install
```

3. Pull lfs

```bash
git lfs pull
```

## Local

1. Install Packages

```bash
pip3 install -r requirements.txt
```

2. Run Development Server

```bash
fastapi dev app.py
```

## Docker

Build Image

```bash
docker build -t pixolve-backend .
```

Run Docker Container

```bash
docker run -p 8000:8000 pixolve-backend
```
