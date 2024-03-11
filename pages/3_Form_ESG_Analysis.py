import streamlit as st
from sec_utils import get_filings, generate_presigned_url
from pdfminer.high_level import extract_text
import boto3
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import os
import nltk
import requests
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import torch

def get_text_from_pdf(file_path):
    text = extract_text(file_path)
    return text

st.title('Form ESG Analysis')

st.markdown('''
This page allows you to analyze the ESG content of a company's SEC filings.
Select a company and a form to analyze, and click the 'Analyze' button to view the results.

If you don't see the form you're looking for, you can track new filings on the 'Track Filings' page.
''')

model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg-9-categories', num_labels=9)
tokenizer = BertTokenizer.from_pretrained(
    'yiyanghkust/finbert-esg-9-categories',
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

class_labels = ['Climate Change', 'Natural Capital', 'Pollution & Waste',
                'Human Capital', 'Product Liability', 'Community Relations',
                'Corporate Governance', 'Business Ethics & Values', 'Non-ESG']

# This page has a dropdown called 'ticker' that allows the user to select a company to analyze.
# The ticker dropdown is populated with values from dynamodb.
# Once a user selects a dropdown value, a new dropdown called 'Form' appears.
# The 'Form' dropdown is populated with values f'{filedAt} - {formType}' from dynamodb.
# A button called 'Analyze' appears after the user selects a value from the 'Form' dropdown.

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('filings')
filings_df = pd.DataFrame(table.scan()['Items'])
ticker = st.selectbox(
    'Select a company to analyze:',
    set(filings_df['ticker']),
    index=0
)

# Filter for items with the selected ticker
filtered_filings_df = filings_df[filings_df['ticker'] == ticker]

# Create a new column called 'form' that concatenates the 'filedAt' and 'formType' columns
filtered_filings_df['form'] = filtered_filings_df['filedAt'] + ' - ' + filtered_filings_df['formType']

# Sort the dataframe by 'filedAt' in descending order
filtered_filings_df = filtered_filings_df.sort_values(by='filedAt', ascending=False)

# Create a new dropdown called 'Form' that is populated with the 'form' column
form = st.selectbox(
    'Select a form to analyze:',
    filtered_filings_df['form'],
    index=0
)

# Create a button called 'Analyze' that appears after the user selects a value from the 'Form' dropdown
if st.button('Analyze'):
    # Retrieve the selected filing from dynamodb
    selected_filing = filtered_filings_df[filtered_filings_df['form'] == form].to_dict(orient='records')[0]
    # Retrieve the text of the filing from S3

    # download the filing from S3
    s3_client = boto3.client('s3')
    object_key = selected_filing['objectKey']
    filename = selected_filing['s3Url'].split('/')[-1]

    # generate a presigned URL for the S3 object
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': os.environ['S3_BUCKET_NAME'], 'Key': object_key},
        ExpiresIn=600
    )

    # download the filing from the presigned URL
    progress_bar = st.progress(0, text='Downloading filing...')
    response = requests.get(presigned_url)

    # response.content contains the PDF of the filing. Need to extract the text from the PDF.
    with open('temp.pdf', 'wb') as f:
        f.write(response.content)
    progress_bar.progress(0.001, text='Extracting text from PDF...')
    text = get_text_from_pdf('temp.pdf')
    os.remove('temp.pdf')

    # use nltk to split the text into sentences
    sentences = nltk.sent_tokenize(text)

    ESG_counts = {
        'Climate Change': 0,
        'Natural Capital': 0,
        'Pollution & Waste': 0,
        'Human Capital': 0,
        'Product Liability': 0,
        'Community Relations': 0,
        'Corporate Governance': 0,
        'Business Ethics & Values': 0,
        'Non-ESG': 0
    }

    progress_text = 'Analyzing sentences...'
    progress_bar.progress(0.002, text=progress_text)
    sentences_by_type = {label: [] for label in class_labels}
    # use finbert to classify each sentence
    for i, sentence in enumerate(sentences):
        if len(sentence) < 10:
            continue
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        result = model(**inputs)
        logits = result.logits
        class_probs = torch.nn.functional.softmax(logits, dim=1).cpu().detach().numpy().flatten()
        predicted_label = torch.argmax(logits, dim=1)
        # convert object like tensor([2], device='cuda:0') to int
        predicted_label = predicted_label.item()
        # convert the predicted label to a string with the .map method
        index_to_label = {i: label for i, label in enumerate(class_labels)}
        
        label = index_to_label[predicted_label]
        probability = class_probs[predicted_label]

        if (label!='Non-ESG') & (probability > 0.98): # only count sentences with high confidence
            ESG_counts[label] += 1
            sentences_by_type[label].append(sentence)
        else:
            ESG_counts['Non-ESG'] += 1
            sentences_by_type['Non-ESG'].append(sentence)

        if i % 50 == 0:
            progress_bar.progress(i/len(sentences), text=progress_text + f' {i}/{len(sentences)}')
    progress_bar.empty()

    st.write('ESG Analysis Results:')
    st.write(ESG_counts)
    st.write('Total sentences:', len(sentences))
    st.write('Total ESG sentences:', sum(ESG_counts.values()) - ESG_counts['Non-ESG'])
    st.write('Total Non-ESG sentences:', ESG_counts['Non-ESG'])

    # Display some examples of the different types of ESG sentences
    st.write('Examples of Climate Change sentences:')
    st.write(sentences_by_type['Climate Change'][:5])
    st.write('Examples of Natural Capital sentences:')
    st.write(sentences_by_type['Natural Capital'][:5])
    st.write('Examples of Pollution & Waste sentences:')
    st.write(sentences_by_type['Pollution & Waste'][:5])
    st.write('Examples of Human Capital sentences:')
    st.write(sentences_by_type['Human Capital'][:5])
    st.write('Examples of Product Liability sentences:')
    st.write(sentences_by_type['Product Liability'][:5])
    st.write('Examples of Community Relations sentences:')
    st.write(sentences_by_type['Community Relations'][:5])
    st.write('Examples of Corporate Governance sentences:')
    st.write(sentences_by_type['Corporate Governance'][:5])
    st.write('Examples of Business Ethics & Values sentences:')
    st.write(sentences_by_type['Business Ethics & Values'][:5])
    st.write('Examples of Non-ESG sentences:')
    st.write(sentences_by_type['Non-ESG'][:5])

    # Display the ESG counts in a bar chart
    st.bar_chart(pd.Series(ESG_counts).drop('Non-ESG'), use_container_width=True)