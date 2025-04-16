import express from 'express';
import cors from 'cors';
import * as tf from '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const port = process.env.PORT || 3002;

app.use(cors());
app.use(express.json());

let model: use.UniversalSentenceEncoder | null = null;

// Initialize the model
async function initializeModel() {
  try {
    console.log('Loading Universal Sentence Encoder model...');
    model = await use.load();
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Error loading model:', error);
    process.exit(1);
  }
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', modelLoaded: !!model });
});

// Text processing endpoint
app.post('/process', async (req, res) => {
  if (!model) {
    return res.status(503).json({ error: 'Model not loaded' });
  }

  try {
    const { text } = req.body;
    if (!text) {
      return res.status(400).json({ error: 'Text is required' });
    }

    const embeddings = await model.embed(text);
    const result = await embeddings.array();
    
    res.json({
      embeddings: result[0],
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error processing text:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Start the server
initializeModel().then(() => {
  app.listen(port, () => {
    console.log(`ML Service running on port ${port}`);
  });
});
