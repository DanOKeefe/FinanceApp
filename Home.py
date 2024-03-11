import streamlit as st

st.title("SEC Filings Explorer")

import numpy as np

import pandas as pd

# generate a sample dataframe and display it with st.dataframe()
data = np.random.randn(10, 20)
df = pd.DataFrame(data, columns=[f'col{i}' for i in range(20)])

# instead, make the dataframe editable with st.data_editor()
edited_df = st.data_editor(df)


st.markdown("""
#### What are SEC Filings?

The U.S. Securities and Exchange Commission (SEC) requires public companies to file financial reports and other documents. These filings are available to the public and can be used to track a company's financial performance, operations, and future plans. Types of filing include:

#### 10-K: Annual Reports
- Dive deep into a company's financial performance, risks, governance, operations, and more.
#### 10-Q: Quarterly Reports
- Provides a summary of a company's financial performance for the quarter.
- Useful for tracking a company's performance between annual reports.
#### 8-K: Current Reports
- Provides updates on important events such as acquisitions, changes in management, bankruptcy, and more.
- Must be filed within 4 business days of the event.
#### Forms 3, 4, and 5: Insider Trading Reports
- Filed by company insiders to report their trading activity.
- Form 3: Initial filing
- Form 4: Changes in ownership
- Form 5: Annual summary of changes in ownership
#### Proxy Statements
- Provides information about proposals to be voted on at the annual shareholder meeting.
- Includes information about executive compensation, board members, and more.
#### Schedule 13D and 13G: Beneficial Ownership Reports
- Filed by investors who acquire 5% or more of a company's shares.
- Schedule 13D: Active investors
- Schedule 13G: Passive investors
- Used to track changes in ownership and investor sentiment.
- Gives investors warning of potential takeover attempts.
#### Form 144: Insider Trading
- Filed by company insiders to report the sale of restricted stock.
#### Form 20-F: Annual Reports for Foreign Companies
- Similar to the 10-K but filed by foreign companies that trade on U.S. exchanges.
""")