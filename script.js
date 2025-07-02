import {MnistData} from './data.js';

async function showExamples(data) {
  // Create a container in the visor
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];

  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });

    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

function getNewModel() {
  const model = tf.sequential();

  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;

  // In the first layer of our convolutional neural network we have
  // to specify the input shape. Then we specify some parameters for
  // the convolution operation that takes place in this layer.
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  // Repeat another conv2d + maxPooling stack.
  // Note that we have more filters in the convolution.
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten());

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_OUTPUT_CLASSES = 10;
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));


  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

async function train(model, data) {
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = {
    name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 55000;
  const TEST_DATA_SIZE = 10000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [
      d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [
      d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks
  });
}

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}


async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: classNames});

  labels.dispose();
}

const initDrawingCanvas = () => {
  const canvas = document.getElementById('drawing-canvas');
  const ctx = canvas.getContext('2d');

  // Set white background
  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.lineWidth = 10;
  ctx.lineCap = 'round';
  ctx.strokeStyle = '#000000';

  let isDrawing = false;

  canvas.addEventListener('mousedown', () => {
    isDrawing = true;
    ctx.beginPath();
  });

  canvas.addEventListener('mousemove', (event) => {
    if (!isDrawing) return;
    ctx.lineTo(event.offsetX, event.offsetY);
    ctx.stroke();
  });

  canvas.addEventListener('mouseup', () => {
    isDrawing = false;
    ctx.closePath();
  });

  canvas.addEventListener('mouseout', () => {
    isDrawing = false;
    ctx.closePath();
  });

  document.getElementById('clear-button').addEventListener('click', clearCanvas);
};

function clearCanvas() {
  const canvas = document.getElementById('drawing-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Reset white background
  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

const statusMap = {
  STATUS_READY: 'Draw a digit in the box below and click "Predict" to see the model\'s prediction.',
  STATUS_LOADING: 'Loading data...',
  STATUS_TRAINING: 'Training the model...',
}

function setStatus(status) {
  document.getElementById('status-text').innerText = statusMap[status];
}

function visualizePreprocessedInput(inputTensor) {
  const canvas = document.getElementById('preprocessed-canvas');
  tf.browser.toPixels(inputTensor.squeeze(), canvas);
}

function predictDrawing(model) {
  const canvas = document.getElementById('drawing-canvas');
  const ctx = canvas.getContext('2d');

  const inputTensor = tf.tidy(() => {
    // Convert canvas to tensor (grayscale)
    let img = tf.browser.fromPixels(canvas, 1);

    // Resize to 28x28 pixels
    img = tf.image.resizeBilinear(img, [28, 28]);

    // Normalize to [0, 1] range first
    img = img.div(255.0);

    // Invert colors (canvas is black on white, MNIST is white on black)
    // For normalized values: 1 - pixel_value
    img = tf.scalar(1.0).sub(img);

    // Add batch dimension to get [1, 28, 28, 1]
    img = img.expandDims(0);

    return img;
  });

  const prediction = model.predict(inputTensor);
  const predictedClass = prediction.argMax(-1).dataSync()[0];
  const confidence = tf.max(prediction).dataSync()[0];

  document.getElementById('prediction-result').innerText =
    `Predicted Class: ${classNames[predictedClass]} (Confidence: ${(confidence * 100).toFixed(1)}%)`;

  visualizePreprocessedInput(inputTensor);
  inputTensor.dispose();
}

async function trainNewModel(data) {
  const model = getNewModel();
  await train(model, data);
  await model.save('localstorage://handwritten-numbers-model');

  return model;
}

async function run() {

  const data = new MnistData();
  await data.load();
  await showExamples(data);
  let model;

  const savedModel  = localStorage.getItem("tensorflowjs_models/handwritten-numbers-model/info");

  if (savedModel) {
    setStatus('STATUS_LOADING');
    model = await tf.loadLayersModel('localstorage://handwritten-numbers-model');
  } else {
    setStatus('STATUS_TRAINING');
    model = await trainNewModel(data);
  }

  initDrawingCanvas();
  document.getElementById('predict-button').addEventListener('click', () => predictDrawing(model));
  document.getElementById('retrain-button').addEventListener('click', async () => {
    setStatus('STATUS_TRAINING');
    model = await trainNewModel(data);
    setStatus('STATUS_READY');
  });

  tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model);
  await showAccuracy(model, data);
  await showConfusion(model, data);
  setStatus('STATUS_READY');
}



document.addEventListener('DOMContentLoaded', run);




