const express = require('express');
const bodyParser = require('body-parser');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const app = express();
const port = process.env.PORT || 8080;


app.use(bodyParser.json());


let model;
const loadModel = async () => {
  const modelPath = path.join(__dirname, 'model', 'model.json');
  model = await tf.loadLayersModel(`file://${modelPath}`);
  console.log('Model loaded successfully');
};


loadModel();


app.get('/', (req, res) => {
  res.send('Hello, world!');
});


app.post('/predict', async (req, res) => {
  try {
    const input = req.body.input;
    const inputTensor = tf.tensor4d(input, [1, 150, 150, 3]);
    const prediction = model.predict(inputTensor);
    const predictionData = prediction.arraySync();
    res.json({ prediction: predictionData });
  } catch (error) {
    res.status(500).send(error.message);
  }
});


app.listen(port, () => {
  console.log(`Server berjalan di http://localhost:${port}/`);
});
