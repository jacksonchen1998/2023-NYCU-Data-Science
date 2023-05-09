# Headline Generation

In this homework, we aim to generate news headlines from news bodies.

You should try to improve the headline quality to increase the performance.

**You can use pre-trained language models as initialization. However, fined-tuned models are not allowed !!!**

## Dataset

- `train.json`: a text file with `100,000` json lines, where each line represents an individual row of data as follows:
    - body: news article in string format
    - title: news title in string format
- `test.json`: a text file with `13,762` json lines, where each line represents an individual row of data as follows:
    - body: news article in string format
- `sample_submission.json`: a sample submission file with `13,762` json lines, where each line represents an individual row of data as follows:
    - title: generated news title in string format

## Method

## Evaluation

- ROUGE
- BERTScore

## Reference
- [Huggingface](https://huggingface.co/docs)
- [Transformers](https://huggingface.co/docs/transformers/quicktour)
- [Tokenizers](https://huggingface.co/docs/tokenizers/quicktour)
- [Various pre-trained models](https://huggingface.co/models)
- [Evaluation](https://huggingface.co/docs/evaluate/a_quick_tour)