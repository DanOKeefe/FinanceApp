from pydantic import BaseModel, HttpUrl
from datetime import datetime

class SECFiling(BaseModel):
    id: str
    ticker: str
    formType: str
    accessionNo: str
    cik: str
    companyNameLong: str
    companyName: str
    linkToFilingDetails: HttpUrl
    description: str
    linkToTxt: HttpUrl
    linkToHtml: HttpUrl
    linkToXbrl: str
    filedAt: datetime
    periodOfReport: datetime
    s3Url: HttpUrl
    secDocumentUrl: HttpUrl
    objectKey: str