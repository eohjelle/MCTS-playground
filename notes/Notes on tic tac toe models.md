Tic tac toe is a simple game which serves as a playground for exploring various algorithms and deep learning models.

# Exploration of different models

The simplicity of the game means that we can solve it, turning the problem of training the model (which is done by self-play for AlphaZero) into a supervised learning problem. See ![the notebook for generating training data](applications/tic_tac_toe/notebooks/tic_tac_toe_training_data.ipynb) for details on how the training data is generated. Using this training data, we experimented with various deep learning model architectures.

For tic tac toe we have explored a few different classes of models:

1. Standard MLPs
2. Traditional transformers
3. Transformers with masked simple attention
4. Transformers with dynamic masked simple attention

Here's an overview of our findings.

- All models can achieve perfect play.
- The MLPs train quickly, but need orders of magnitude more parameters than some of our other models for comparable performance.
- Traditional transformers do not perform very well.
- Our best performing model is a transformer with masked simple attention, achieving minimal loss and perfect play with ~14k parameters.
- Transformers with dynamic masked simple attention. The smolgen-like mask, which depends on the input state, adds extra parameters, and it's difficult to prevent the model from overfitting. even with only around ~20k parameters. The problem is that the training set only contains around 5k samples, but it's a good sign that we don't need many parameters.
