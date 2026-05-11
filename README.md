This project focuses on building a Question Answering (QA) system for low-resource languages using the mT5-small multilingual transformer model. The goal is to enable accurate question answering even when limited annotated data is available.

The system leverages the multilingual capabilities of mT5-small to understand questions and generate relevant answers from given contexts. It is particularly useful for languages that lack large-scale QA datasets. The project demonstrates data preprocessing, fine-tuning, and evaluation of a transformer-based QA model using Python and deep learning libraries.

This work highlights the effectiveness of transfer learning and multilingual models in addressing challenges faced in low-resource NLP settings.

mT5 mT5-small is a lightweight version of the multilingual Text-to-Text Transfer Transformer (mT5) model developed by Google.
It is mainly used for:

        Machine Translation
        Question Answering
        Text Summarization
        Text Classification
        Cross-lingual NLP tasks
        Key Features of mT5-small
        Supports 100+ languages
        


Simple Python Example (Hugging Face)
      from transformers import MT5Tokenizer, MT5ForConditionalGeneration
      model_name = "google/mt5-small"
      tokenizer = MT5Tokenizer.from_pretrained(model_name)
      model = MT5ForConditionalGeneration.from_pretrained(model_name)
      text = "translate English to Hindi: How are you?"
      inputs = tokenizer(text, return_tensors="pt")
      outputs = model.generate(**inputs)
      result = tokenizer.decode(outputs[0], skip_special_tokens=True)
      print(result)
