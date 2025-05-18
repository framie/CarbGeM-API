import express, { Request, Response } from 'express';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import dotenv from 'dotenv';

dotenv.config();  // Load env vars before anything else

const app = express();
const PORT = Number(process.env.PORT) || 3000;
const pythonPath = process.env.PYTHON_PATH;
const uploadDir = path.join(__dirname, '..', 'uploads');
const outputDir = path.join(__dirname, '..', 'output');

[uploadDir, outputDir].forEach((dir) => {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
        console.log(`Created directory: ${dir}`);
    }
});

const storage = multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, uploadDir),
    filename: (_req, file, cb) => cb(null, file.originalname)
});
const upload = multer({ storage });

app.post('/analyze', upload.single('image'), async (req: Request, res: Response) => {
    if (!req.file) {
        res.status(400).send('No file uploaded.');
        return;
    }

    const filePath = path.join(uploadDir, req.file.originalname);
    const outputImagePath = path.join(outputDir, req.file.originalname);

    // If image already exists, skip storage
    if (!fs.existsSync(filePath)) {
        fs.renameSync(req.file.path, filePath);
    }

    // If processed output already exists, return response immediately
    if (fs.existsSync(outputImagePath)) {
        res.sendFile(outputImagePath);
        return;
    }

    // Call the Python script to analyze image for Zones of Inhibition
    const python = spawn(pythonPath || 'python3', [
        path.join(__dirname, '..', 'analyze.py'),
        filePath,
        outputImagePath
    ]);

    python.stderr.on('data', (data) => console.error(`stderr: ${data}`));

    python.on('close', (code) => {
        if (code !== 0) {
            res.status(500).send('Python script failed.');
        } else if (fs.existsSync(outputImagePath)) {
            res.sendFile(outputImagePath);
        } else {
            res.status(500).send('Output image not found.');
        }
    });
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on port ${PORT}`);
});
