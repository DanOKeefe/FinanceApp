The goal of this project is to extract valuable insights into a company given their SEC filings and transcripts of conference calls.

Can we tell a story about a company and its management over time?

### Create .env file and define the following environment variable

```
#OPENAI_API_KEY="" not used
SEC_API_KEY="
S3_BUCKET="s3://<bucket_name>"
S3_BUCKET_NAME="<bucket_name>"
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
AWS_DEFAULT_REGION=""
```

The SEC_API_KEY should be used from your account on https://sec-api.io/

Run the app with

```
streamlit run Home.py
```
