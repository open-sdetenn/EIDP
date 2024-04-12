EIDP is code for the micro-paper 'Enhanced Inverse Dynamics Prediction for Robot Control Using Deep Residual Networks' (aka. a beefed up IDM)
[paper](https://wandb.ai/mikusdevr/IDM/reports/Enhanced-Inverse-Dynamics-Prediction-for-Robot-Control-Using-Deep-Residual-Networks--Vmlldzo3NDczNDc1)

most recent model is @ [https://huggingface.co/opensdetenn/resnet18_linear_v1](https://huggingface.co/opensdetenn/resnet18_linear_v1)

colab coming soon, use current code. download the weights you want and infer.

running this is rather self-explanitory. if someone on this project wants to commit to a proper read-me, go ahead.

Just want to state the architecture; It's literally just the ResNet-18 architecture, finetuned on the driving set, with each of the vectors being attached to a linear neuron. Nothing insane, just efficient.
