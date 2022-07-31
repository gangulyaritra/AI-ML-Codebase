# Implementation of a Contextual Chatbot in PyTorch.

Simple Chatbot Implementation with PyTorch.

- The implementation should be easy to follow for beginners and provide a basic understanding of chatbots.
- The implementation is straightforward, with a Feed-Forward Neural Network consisting of 2 hidden layers.
- Customization for your use case is super easy. Just modify `intents.json` with possible patterns and responses and re-run the training.

## Watch the YouTube Tutorial.
[![Alt text](https://img.youtube.com/vi/RpWeNzfSUHw/hqdefault.jpg)](https://www.youtube.com/watch?v=RpWeNzfSUHw&list=PLqnslRFeH2UrFW4AUgn-eY37qOAWQpJyg)

## Usage.
Run
```console
python train.py
```
This will dump `data.pth` file. And then run
```console
python chat.py
```

# Chatbot Deployment with Flask and JavaScript.

This application gives 2 deployment options:

- Deploy within Flask app with jinja2 template.
- Serve only the Flask prediction API. The used HTML and javascript files can be included in any Frontend application (with only a slight modification) and can run completely separate from the Flask App.

[![Alt text](https://img.youtube.com/vi/a37BL0stIuM/hqdefault.jpg)](https://www.youtube.com/watch?v=a37BL0stIuM)