from sec_client import SECAPIClient
import os
import boto3
from pdfminer.high_level import extract_text
from datetime import datetime
import uuid
import requests

from models.models import SECFiling
from dotenv import load_dotenv

from sec_api import QueryApi

load_dotenv()

def write_to_s3(bucket_name, file_name, object_name):
    s3_client = boto3.client('s3')
    s3_client.upload_file(file_name, bucket_name, object_name)
    s3_url = f'https://{bucket_name}.s3.amazonaws.com/{object_name}'

    return s3_url

def write_to_filings_table(sec_filing: SECFiling):
    """
    Write metadata about a document to a documents collection in DynamoDB
    Accepts an SECFiling object and writes it to the documents table
    """
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('filings')
    item = sec_filing.dict()
    for key, value in item.items():
        if isinstance(value, datetime):
            item[key] = value.strftime('%Y-%m-%d %H:%M:%S')
        elif not isinstance(value, str):
            item[key] = str(value)
            
    table.put_item(Item=item)

def get_text_from_pdf(file_path):
    text = extract_text(file_path)

    return text

def get_filings(ticker, start_date, end_date, form_type, limit=10000):
    """
    state_date and end_date filter on the filedAt field. Should be in the format "YYYY-MM-DD"
    form_type is the type of filing to retrieve. For example, "10-Q" or "10-K" https://sec-api.io/list-of-sec-filing-types
    """
    # Retrieve filings metadata
    # iterate through the filings and download the filing as a PDF
    # use get_text_from_pdf to extract the text from the PDF
    # upload the PDF to S3
    # create a new SECFiling object and write it to the database

    sec_client = QueryApi(api_key=os.environ['SEC_API_KEY'])

    query = {
        "query": { "query_string": { 
            "query": f"ticker:{ticker} AND filedAt:[{start_date} TO {end_date}] AND formType:\"{form_type}\"",
            "time_zone": "America/New_York"
        } },
        "from": "0",
        "size": f"{limit}",
        "sort": [{ "filedAt": { "order": "desc" } }]
    }

    headers = {'Authorization': os.environ['SEC_API_KEY']}
    filings = sec_client.get_filings(query)
    for filing in filings['filings']:
        cik = filing['cik']
        accession_number = filing['accessionNo'].replace('-', '')
        filename = filing['linkToTxt'].split('/')[-1]

        # iterate through the documentFormatFiles and identify the document that represents the form
        # and we want the documentUrl field from this document. This ithe the url that should be sent to the pdf generator.
        for doc in filing['documentFormatFiles']:
            if doc['type'] == form_type:
                doc_url = doc['documentUrl']
                break

        if not doc_url:
            print(f'Could not find documentUrl for {filing["accessionNo"]}')
            continue

        # download the filing as a PDF
        url = f'https://api.sec-api.io/filing-reader/?token={os.environ["SEC_API_KEY"]}&type=pdf&url={doc_url}'
        response = requests.get(url)
        filename = f'{filing["accessionNo"]}.pdf'
        with open(filename, 'wb') as f:
            f.write(response.content)

        object_name = f'{filing["ticker"]}/{filing["formType"]}/{filing["filedAt"]}/{filename}'
        s3_url = write_to_s3(bucket_name='dto-sec-filings', file_name=filename, object_name=object_name)
        os.remove(filename)
        sec_filing = SECFiling(
            id = str(uuid.uuid4()),
            accessionNo=filing['accessionNo'],
            cik=filing['cik'],
            ticker=filing['ticker'],
            companyName=filing['companyName'],
            companyNameLong=filing['companyNameLong'],
            formType=filing['formType'],
            periodOfReport=filing['periodOfReport'],
            description=filing['description'],
            filedAt=filing['filedAt'],
            linkToTxt=filing['linkToTxt'],
            linkToHtml=filing['linkToHtml'],
            linkToXbrl=filing['linkToXbrl'],
            linkToFilingDetails=filing['linkToFilingDetails'],
            s3Url=s3_url,
            secDocumentUrl=doc_url,
            objectKey=object_name
        )
        write_to_filings_table(sec_filing)

def generate_presigned_url(bucket_name, object_key, expiration=3600):
    """
    Generate a presigned URL for downloading a file from S3.

    Parameters:
        - bucket_name (str): The name of the S3 bucket.
        - object_key (str): The key of the object (file) in the S3 bucket.
        - expiration (int): The time, in seconds, until the presigned URL expires. Default is 3600 seconds (1 hour).

    Returns:
        str: The presigned URL for downloading the file.
    """
    s3 = boto3.client('s3')
    
    # Generate a presigned URL for the S3 object
    presigned_url = s3.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': bucket_name,
            'Key': object_key
        },
        ExpiresIn=expiration
    )

    return presigned_url