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
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import numpy as np
import torch
import altair as alt
import matplotlib.pyplot as plt

def get_text_from_pdf(file_path):
    text = extract_text(file_path)
    return text

st.title('Forward Statement Sentiment Analysis')

st.markdown('''
This page allows you to analyze the sentiment of forward looking statements in SEC filings.
Select a company and a form to analyze. If the form has already been analyzed, the results will be displayed.
If the company has 3 or more 10-K filings that have been analyzed, the trend in sentiment of forward looking statements over time will be displayed.

If you don't see the form you're looking for, you can track new filings on the 'Track Filings' page.
''')

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

# Check if this form has already been analyzed
# If it has, display the results of the analysis
def has_forward_looking_statement_sentiments(form_id):
    response = table.get_item(
        Key={
            'id': form_id
        }
    )
    if 'forwardStatementSentiment' not in response['Item']:
        return False
    else:
        return True
    forward_looking_statement_sentiments = response['Item']['forwardStatementSentiment']

form_id = filtered_filings_df[filtered_filings_df['form'] == form]['id'].values[0]
if has_forward_looking_statement_sentiments(form_id):
    forward_looking_statement_sentiments = table.get_item(
        Key={
            'id': form_id
        }
    )['Item']['forwardStatementSentiment']
    sentiment_counts = {
        'positive': int(forward_looking_statement_sentiments['positiveCount']),
        'negative': int(forward_looking_statement_sentiments['negativeCount']),
        'neutral': int(forward_looking_statement_sentiments['neutralCount'])
    }
    x_axis_labels = ['positive', 'negative', 'neutral']

    bar_chart = alt.Chart(pd.DataFrame(sentiment_counts.items(), columns=['sentiment', 'count'])).mark_bar().encode(
        x='sentiment',
        y='count',
        color=alt.Color('sentiment', scale=alt.Scale(domain=['positive', 'negative', 'neutral'], range=['green', 'red', 'blue']))
    ).properties(
        title='Counts of Positive, Negative, and Neutral Forward Looking Statements'
    )

    st.altair_chart(bar_chart.encode(x=alt.X('sentiment', type='nominal')), use_container_width=True)
    st.write('This form has already been analyzed. The results are displayed above.')

    # Filter for 10-K filings
    # filtered_filings_df has already been filtered for the selected ticker
    filtered_10k_filings_df = filtered_filings_df[filtered_filings_df['formType'] == '10-K']

    # Check to see if there are 3 or more 10-K filings that have been analyzed
    analyzed_10k_filings = []
    for form_id in filtered_10k_filings_df['id']:
        if has_forward_looking_statement_sentiments(form_id):
            analyzed_10k_filings.append(form_id)

    if len(analyzed_10k_filings) >= 3:
        st.write('This company has 3 or more 10-K filings that have been analyzed. The trend in sentiment of forward looking statements over time is displayed below.')
        # Retrieve the forward looking statement sentiments for each 10-K filing
        forward_looking_statement_sentiments = []
        period_of_reports = []
        for form_id in analyzed_10k_filings:
            item = table.get_item(
                Key={
                    'id': form_id
                }
            )['Item']
            forward_looking_statement_sentiments.append(item['forwardStatementSentiment'])
            period_of_reports.append(item['periodOfReport'])

        trend_data = []
        for period_of_report, forward_looking_statement_sentiment in zip(period_of_reports, forward_looking_statement_sentiments):
            trend_data.append({
                'periodOfReport': period_of_report,
                'positiveCount': forward_looking_statement_sentiment['positiveCount'],
                'negativeCount': forward_looking_statement_sentiment['negativeCount'],
                'neutralCount': forward_looking_statement_sentiment['neutralCount']
            })

        trend_df = pd.DataFrame(trend_data)
        trend_df = trend_df.melt(id_vars='periodOfReport', value_vars=['positiveCount', 'negativeCount', 'neutralCount'], var_name='sentiment', value_name='count')
        trend_df = trend_df.sort_values(by=['periodOfReport', 'sentiment'])
        

        # example of what this trend_df looks like in markdown
        # | periodOfReport | sentiment | count |
        # | -------------- | --------- | ----- |
        # | 2021-01-01     | negativeCount | 5 |
        # | 2021-01-01     | neutralCount | 10 |
        # | 2021-01-01     | positiveCount | 15 |
        # | 2020-01-01     | negativeCount | 3 |
        # | 2020-01-01     | neutralCount | 8 |

        fig, ax = plt.subplots()
        for sentiment in ['positiveCount', 'negativeCount', 'neutralCount']:
            sentiment_df = trend_df[trend_df['sentiment'] == sentiment]
            # make the positive counts green, the negative counts red, and the neutral counts blue
            color = 'green' if sentiment == 'positiveCount' else 'red' if sentiment == 'negativeCount' else 'blue'
            ax.plot(sentiment_df['periodOfReport'], sentiment_df['count'], label=sentiment, color=color)
        ax.set_xlabel('Period of Report')
        ax.set_ylabel('Count')
        ax.set_title('Trend in Sentiment of Forward Looking Statements Over Time')
        ax.legend()
        # adjust the x axis labels so that they are not overlapping
        # make them at an angle, close to vertical
        plt.xticks(rotation=80)
        st.pyplot(fig)

        st.write(trend_df)

else:
    st.write('This form has not been analyzed yet. Click the button below to analyze the forward looking statements in this form.')

    # Create a button called 'Analyze' that appears after the user selects a value from the 'Form' dropdown

    if st.button('Analyze'):
        # Specific-FLS , Non-specific FLS, or Not-FLS.
        model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-fls',num_labels=3)
        tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-fls')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

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
        response = requests.get(presigned_url, stream=True)

        with open(filename, 'wb') as f:
            f.write(response.content)
        
        # extract the text from the filing
        text = get_text_from_pdf(filename)
        # identify forward looking statements
        forward_looking_statements = []
        forward_looking_statement_indices = []
        sentences = nltk.sent_tokenize(text)

        for i, sentence in enumerate(sentences):
            # 512 is max input sequence length for this model
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs.to(device)
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(predictions).item()
            label_map = {0: 'Not-FLS', 1: 'Non-specific FLS', 2: 'Specific-FLS'}
            label = label_map[prediction]
            if label == 'Specific-FLS':
                forward_looking_statements.append(sentence)
                forward_looking_statement_indices.append(i)

        # analyze the sentiment of each forward looking statement
        del model
        sentiment_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
        sentiment_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        sentiment_model.to(device)

        forward_looking_statement_sentiments = []
        for forward_looking_statement in forward_looking_statements:
            inputs = sentiment_tokenizer(forward_looking_statement, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs.to(device)
            outputs = sentiment_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(predictions).item()
            label_map = {0: 'neutral', 1: 'positive', 2: 'negative'}
            label = label_map[prediction]
            forward_looking_statement_sentiments.append(label)

        # store the results of the forward looking statement sentiment analysis in the filings table
        sentiment_counts = {
            'positive': forward_looking_statement_sentiments.count('positive'),
            'negative': forward_looking_statement_sentiments.count('negative'),
            'neutral': forward_looking_statement_sentiments.count('neutral')
        }

        forward_looking_statement_sentiments = {
            'forwardStatements': [
                {
                    'text': forward_looking_statement,
                    'sentiment': sentiment
                }
                for forward_looking_statement, sentiment in zip(forward_looking_statements, forward_looking_statement_sentiments)
            ],
            'positiveCount': sentiment_counts['positive'],
            'negativeCount': sentiment_counts['negative'],
            'neutralCount': sentiment_counts['neutral']
        }

        # update the filings table with the results of the forward looking statement sentiment analysis
        # This will be an update to only the selected filing.
        # I will add a new column to the filings table called 'forwardStatementSentiment'.
        # This column will contain the results of the forward statement sentiment analysis.

        form_id = selected_filing['id']
        table.update_item(
            Key={
                'id': form_id
            },
            UpdateExpression='SET forwardStatementSentiment = :forwardStatementSentiment',
            ExpressionAttributeValues={':forwardStatementSentiment': forward_looking_statement_sentiments}
        )

        # delete the PDF from the local filesystem
        os.remove(filename)
        st.write('Forward looking statement sentiment analysis complete!')

        forward_looking_statement_sentiments = table.get_item(
            Key={
                'id': form_id
            }
        )['Item']['forwardStatementSentiment']
        sentiment_counts = {
            'positive': int(forward_looking_statement_sentiments['positiveCount']),
            'negative': int(forward_looking_statement_sentiments['negativeCount']),
            'neutral': int(forward_looking_statement_sentiments['neutralCount'])
        }
        x_axis_labels = ['positive', 'negative', 'neutral']

        bar_chart = alt.Chart(pd.DataFrame(sentiment_counts.items(), columns=['sentiment', 'count'])).mark_bar().encode(
            x='sentiment',
            y='count',
            color=alt.Color('sentiment', scale=alt.Scale(domain=['positive', 'negative', 'neutral'], range=['green', 'red', 'blue']))
        ).properties(
            title='Counts of Positive, Negative, and Neutral Forward Looking Statements'
        )