# To start training a new model on provided datasets, e.g., Digit, run:
python train.py --dataset uci

# To test the trained model, e.g., Digit, run:
python test.py --dataset uci

# Due to the divergence based loss may fail into inferior local optima, there may be some fluctuations in the running results, which can be improved by more attempts.

# Acknowledgements
We thank the Pytorch implementation on SiMVC https://github.com/DanielTrosten/mvc.