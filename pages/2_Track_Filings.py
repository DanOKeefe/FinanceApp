import streamlit as st
from sec_utils import get_filings, generate_presigned_url
import boto3
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv()

st.title('Track Filings')
st.write('Welcome to the Track Filings page! This page allows you to track SEC filings for publicly traded companies.')
st.write('Enter the ticker, start date, end date, and form type to track filings.')

ticker = st.text_input('Enter ticker:')
start_date = st.date_input(
    'Enter earliest date of filing:',
    value=pd.to_datetime('2000-01-01'),
    min_value=pd.to_datetime('1970-01-01'),
    max_value=datetime.now().date()
)
end_date = st.date_input(
    'Enter latest date of filing:',
    value=datetime.now().date(),
    min_value=pd.to_datetime('1970-01-01'),
    max_value=datetime.now().date()
)
form_type = st.text_input('Enter form type (e.g. 10-K):', value='10-K')
limit = st.number_input('Enter the number of filings to retrieve:', min_value=1, max_value=200, value=1)

if st.button('Track Filings'):
    st.write('Retrieving filings...')
    get_filings(ticker, start_date, end_date, form_type, limit)
    st.write('Filings tracked successfully!')
    st.write('PDFs of the filings have been saved to S3 and metadata about the filings has been saved to DynamoDB. You can view the metadata in the table below.')

    # Retrieve metadata about the filings from dynamodb and
    # display it in a dataframe
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table('filings')

    query_params = {
        'ticker': ticker,
        'formType': form_type
    }

    response = table.scan(
        FilterExpression=' AND '.join([f'{k} = :{k}' for k in query_params.keys()]),
        ExpressionAttributeValues={f':{k}': v for k, v in query_params.items()}
    )
    sorted_results = sorted(response['Items'], key=lambda x: x['filedAt'], reverse=True)

    columns = ['ticker', 'formType', 'periodOfReport', 'companyName', 'secDocumentUrl', 's3Url', 'objectKey']
    filings_df = pd.DataFrame(sorted_results)[columns]
    filings_df['periodOfReport'] = pd.to_datetime(filings_df['periodOfReport']).dt.strftime('%Y-%m-%d')
    edited_df = st.data_editor(
        # don't display the s3Url or objectKey column
        filings_df[['ticker', 'formType', 'periodOfReport', 'companyName', 'secDocumentUrl']],
        column_config={
            'secDocumentUrl': st.column_config.LinkColumn(
                'SEC Filing Link',
                help='Link to SEC document',
                display_text='Link'
            )
        },
        hide_index=True
    )

    st.markdown('### Download Filings')
    # s3_download_urls = [f'<a href="s3://{os.environ["S3_BUCKET"]}/{row["ticker"]}/{row["formType"]}/{row["s3Url"].split("/")[-1]}">Download from S3</a>' for index, row in filings_df.iterrows()]
    for index, row in filings_df.iterrows():
        presigned_url = generate_presigned_url(os.environ['S3_BUCKET_NAME'], object_key=row['objectKey'], expiration=600)
        st.markdown(f'<a href="{presigned_url}">Download {row["ticker"]} {row["formType"]} on {row["periodOfReport"]} from S3</a>', unsafe_allow_html=True)
        #st.markdown(f'<a href="{row["s3Url"]}">Download {row["ticker"]} {row["formType"]} on {row["periodOfReport"]} from S3</a>', unsafe_allow_html=True)