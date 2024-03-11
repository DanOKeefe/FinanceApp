import streamlit as st
import boto3
import pandas as pd
from dotenv import load_dotenv
from sec_utils import generate_presigned_url
import os

load_dotenv()

# Connect to DynamoDB
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('filings')

# Get user input for filtering and sorting
#ticker_filter = st.text_input('Enter ticker:')
# make the ticker_filder be a dropdown.
# possible values should be each unique ticker in the filings table
tickers = table.scan(ProjectionExpression='ticker')['Items']
if not tickers:
    st.write('No tickers found in the filings table')

unique_tickers = set([ticker['ticker'] for ticker in tickers])
ticker_filter = st.selectbox(
    'Filter by ticker:', 
    unique_tickers,
    index=0
)

form_type_filter = st.text_input('Filter by form type:')

# Query DynamoDB based on user input
query_params = {}
if ticker_filter:
    query_params['ticker'] = ticker_filter
if form_type_filter:
    query_params['formType'] = form_type_filter

if query_params:
    response = table.scan(FilterExpression=' AND '.join([f'{k} = :{k}' for k in query_params.keys()]),
                      ExpressionAttributeValues={f':{k}': v for k, v in query_params.items()})
else:
    response = table.scan()

# Take the response['Items'] list and use the SECFiling model for data validation
from models.models import SECFiling
filings = [SECFiling(**filing).dict() for filing in response['Items']]

# Sort the results based on filedAt
sorted_filings = sorted(filings, key=lambda x: x['filedAt'], reverse=True)

columns = ['ticker', 'formType', 'periodOfReport', 'companyName', 'secDocumentUrl', 'objectKey']
# Display the results in a dataframe
filings_df = pd.DataFrame(sorted_filings)[columns]
filings_df['periodOfReport'] = pd.to_datetime(filings_df['periodOfReport'])
filings_df['periodOfReport'] = filings_df['periodOfReport'].dt.strftime('%Y-%m-%d')
filings_df['s3DownloadUrl'] = filings_df['objectKey'].apply(lambda x: generate_presigned_url(os.environ['S3_BUCKET_NAME'], x))

edited_df = st.data_editor(
    filings_df[['ticker', 'formType', 'periodOfReport', 'companyName', 'secDocumentUrl', 's3DownloadUrl']],
    column_config={
        'secDocumentUrl': st.column_config.LinkColumn(
            'SEC Filing Link',
            help='Link to SEC document',
            display_text='SEC Link'
        ),
        's3DownloadUrl': st.column_config.LinkColumn(
            'S3 Download',
            help='Link to download the filing from S3',
            display_text='Download from S3'
        )
    },
    hide_index=True,
    width=1500
)