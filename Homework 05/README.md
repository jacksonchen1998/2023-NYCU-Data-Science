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
- `311511052.json`: a sample submission file with `13,762` json lines, where each line represents an individual row of data as follows:
    - title: generated news title in string format

## Method

Using `T5-Small` model and fine-tune it on the dataset.

- [A Full Guide to Finetuning T5 for Text2Text and Building a Demo with Streamlit](https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887)
- [Hugging Face NLP Course Summarization](https://huggingface.co/learn/nlp-course/chapter7/5?fw=pt)

## Reproduce

With the `requirements.txt` file, you can install all the required packages by running the following command:

```bash
pip install -r requirements.txt
```

And with shell script, you can reproduce the result by running the following command:

```bash
bash 311511052.sh
```

## Evaluation

- ROUGE
    - ROUGE-1
    - ROUGE-2
    - ROUGE-L
- BERTScore

## Reference
- [Huggingface](https://huggingface.co/docs)
- [Transformers](https://huggingface.co/docs/transformers/quicktour)
- [Tokenizers](https://huggingface.co/docs/tokenizers/quicktour)
- [Various pre-trained models](https://huggingface.co/models)
- [Evaluation](https://huggingface.co/docs/evaluate/a_quick_tour)