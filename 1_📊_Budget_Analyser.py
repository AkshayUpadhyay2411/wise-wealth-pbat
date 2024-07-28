import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

OPENAI_API = st.secrets["api_keys"]["OPENAI_API"]
os.environ["OPENAI_API_KEY"] = OPENAI_API
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "testing-english-conv"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["api_keys"]["LANGSMITH_API"]
os.environ["GROQ_API_KEY"] = st.secrets["api_keys"]["GROQ_API_KEY"]

# model_name = "gpt-3.5-turbo"
model_name = "gpt-4o-mini"
llm = ChatOpenAI(model_name=model_name)
# llm = ChatGroq(model_name="llama3-70b-8192")

template = """\
You are a professional Chartered Accountant who consults individuals and helps give actionable advice, reviews, and recommendations. I will later provide information about the recently announced Budget 2024-2025 by the Indian government. Using only that information and using every detail of the individual provided to you, you will generate a report using the given format, highlighting key announcements and changes that they need to be aware of, in context of taxation. 


Keep the following in mind: 
1. Follow the instructions provided for each section in the report format.
2. Keep it concise, and based on the information in the original documents. Skip sections in the report where there are no meaningful announcements or changes. 
3. Try to focus more on highlighting new additions in the budget and key changes. 
4. Use the individual's unique details and context to answer and eliminate information that is not related to that individual's provided. 
Keep the language simple to read. Any technical or non-obvious terms should be explained in brief. 


Here are the details provided by the individual. 
Name: {name}
Income source: {income_source}
Investment interests in asset types: {investment_interest_assets}
Investment durations: {investment_duration}
Sectors invested in: {invested_sectors}
Liability types: {liability_types}
Regime selected: {tax_regime}
Plan to start business in any sectors?: {business_sectors}


Use the following format, information provided and individual details provided to generate a report: 
```
## Budget Impact Analysis Report

### 1. Income Taxation
#### Key Announcements
If the individual opts for new tax regime, refer to 'New vs Old Tax Regime' and do the following: 
- Example of difference in savings if the net income is around 20 Lakhs based on new slab rates and deductions. 
If the individual opts for old tax regime, highlight that there are no changes overall, but there are potential new savings in new tax regime. Mention the most important changes in new regime broadly. 
If the individual is a professional who uses presumptine taxation, mention the announcements related to change in presumptive taxation. 

#### Recommendations
- Refer to 'New vs Old Tax Regime' section to provide recommendations. Keep the recommendations broad and within 2 statements. 


### 2. Business Taxation
Skip this section if the user doesn't own any startup or business

#### Key Announcements
- Changes in corporate tax rates.
- Introduction of new incentives for businesses.
- Modifications to existing business-related deductions or exemptions.

#### Recommendations
- Highlight how to leverage new business incentives, if any, to reduce tax liability.


### 3. Capital Gains Taxation
Skip section if there are no investments interests. 
Refer to Capital Gains Tax Changes to get relevant information. 
Only talk about the asset types that are relevant to the details provided by the individual. Information about asset types not provided by the individual should be avoided. 
Explain technical terms such as LTCG, STCG, etc where necessary. Keep the explanation brief. 

#### Key Announcements
For each relevant asset type -
- Changes in capital gains tax rates of the relevant asset types. Mention before and after.
- Introduction of new investment vehicles with tax benefits for the relevant asset types.
- Modifications to the taxation of dividends and interest income from the relevant asset types.

#### Recommendations
- Rebalance your investment portfolio to align with new tax benefits or tax increases. 
- Suggest new investment options (specific asset types) based on new benefits or tax reductions in the budget.
- If the individual invests in short durations, highlight that the recent increase in short term capital gain taxes discourages short term trading, and long term investments may be more beneficial with respect to taxation. 


### Impact on Liabilities
Skip section if there are no liabilities. 
Only mention information about the provided liability types provided by the individual. 

#### Key Announcements
- Changes affecting the taxation of loans and liabilities.
- Introduction of new measures to support individuals with loans.

#### Recommendations
- Review your loan repayment strategy to optimize tax benefits.
- Consider refinancing or restructuring loans if new benefits are favorable.


### 7. Indirect Taxes

#### Key Announcements
- Changes in GST rates on goods and services.
- Introduction of new indirect taxes.
- Modifications to customs duties and excise duties.
- Changes in customs duty on specific goods (e.g., electronics, automobiles, raw materials).

#### Recommendations
- Adjust your budget to accommodate changes in indirect taxes.
- Explore ways to minimize the impact of increased indirect taxes on your expenses.
- Consider altering your purchasing decisions if customs duties on certain goods have changed.


### 8. Impact on Investments

#### Key Announcements
For each sector, provided by the user, mention about government initiatives in the sectors relevant to the individual. Make sure to include any numbers, data, metrics or other specifics available. 

#### Recommendations
Provide specific sector and stock recommendations. Give priority to sectors provided by the individual. Warn the individual to use self assessment before making any stock investments and clearly mention that this is not an investment advice. 
```

Budget 2024-2025 Information(Use this when answering questions):
```
Here is a detailed summary of the unique sectors and their respective highlights from the recent 23rd July Budget announcements in India, along with the impact on investments and stock recommendations, including stock tickers:
### Infrastructure
**Key Highlights:**
- Increased budgetary allocation for capital expenditures with an expected rise of Rs 0.3 trillion, representing a 20% year-over-year growth
- Focus on modernising existing infrastructure with new technology.
- Plans for building new infrastructure projects

**Impact on Investments:**
- Enhanced infrastructure spending is expected to boost companies involved in construction, engineering, and related services
- Increased capital outlay will benefit contractors and suppliers of construction materials.

**Stock Recommendations:**
- Larsen & Toubro (NSE: LT)
- KNR Constructions (NSE: KNRCON)
- PNC Infratech (NSE: PNCINFRA)


### Renewable Energy
**Key Highlights:**
- Enhanced incentives for renewable energy aligned with the 500 GW target by 2030.
- Potential measures include higher viability gap funding for battery storage, offshore wind plants, and solar energy.
- Steps towards rooftop solarisation.

**Impact on Investments:**
- Companies involved in renewable energy production and ancillary services are expected to benefit from increased incentives and funding.
- Positive outlook for firms specialising in battery storage, wind, and solar energy.

**Stock Recommendations:**
- NTPC (NSE: NTPC)
- Tata Power Company Ltd (NSE: TATAPOWER)
- JSW Energy (NSE: JSWENERGY)
- Adani Power (NSE: ADANIPOWER)
- Suzlon Energy (NSE: SUZLON)
- Inox Wind (NSE: INOXWIND)
- Sterling & Wilson (NSE: SWSOLAR)

### Defence
**Key Highlights:**
- Increased allocation for defence capital expenditure.
- Focus on indigenisation to boost domestic manufacturing and reduce dependency on imports.

**Impact on Investments:**
- Companies involved in defence manufacturing and services are expected to benefit from higher capital expenditure and policy focus on indigenization.
- Positive impact on local job creation and technological advancements in the defence sector.

**Stock Recommendations:**
- Bharat Electronics Ltd (NSE: BEL)
- Hindustan Aeronautics Ltd (NSE: HAL)


### FMCG (Fast-Moving Consumer Goods)
**Key Highlights:**
- Potential tax cuts, expansion in tax slabs, and increased limits for tax-saving investments under Section 80C.
- Any increase in excise duty or National Calamity Contingent Duty (NCCD) on cigarettes and tobacco products would negatively impact cigarette manufacturers.

**Impact on Investments:**
- Increased disposable income due to tax cuts would benefit FMCG companies by boosting consumer spending.
- Negative impact on cigarette manufacturers if excise duties are increased.

**Stock Recommendations:**
- Dabur India (NSE: DABUR)
- Hindustan Unilever (NSE: HINDUNILVR)
- Godrej Consumer Products (NSE: GODREJCP)
- Nestle India (NSE: NESTLEIND)
- ITC Ltd (NSE: ITC)
- Godfrey Phillips (NSE: GODFRYPHLP)


### Real Estate and Housing
**Key Highlights:**
- Reintroduction of the interest subsidy under the Credit Linked Subsidy Scheme (CLSS) for urban housing.
- Increased budget allocation to the PMAY scheme for building rural and urban houses.

**Impact on Investments:**
- Companies involved in affordable housing and construction are expected to benefit from the reintroduction of interest subsidies and increased budget allocations.
- Positive impact on lenders specializing in affordable housing.

**Stock Recommendations:**
- Aavas Financiers (NSE: AAVAS)
- Home First Finance Company India (NSE: HOMEFIRST)
- Macrotech Developers Ltd (NSE: LODHA)
- Sunteck Realty Ltd (NSE: SUNTECK)
- UltraTech Cement (NSE: ULTRACEMCO)
- Ambuja Cements (NSE: AMBUJACEM)
- Dalmia Bharat (NSE: DALBHARAT)

### Healthcare Sector
**Key Highlights:**
- **Total Healthcare Expenditure**: Increased from ₹79,221 crore in 2023-24 to ₹90,171 crore in 2024-25.
- **PMABHIM**: Allocation for Pradhan Mantri Ayushman Bharat Health Infrastructure Mission increased from ₹2,100 crore to ₹4,108 crore.
- **Ayushman Bharat-PMJAY**: Allocation increased from ₹7,200 crore to ₹7,500 crore.
- **PLI Scheme**: Allocation increased from ₹4,645 crore to ₹6,200 crore.
- **Biotechnology R&D**: Allocation increased from ₹500 crore to ₹1,100 crore.
- **Cervical Cancer Vaccination**: Introduction of vaccination programs for girls aged 9-14.
- **Maternal and Child Health**: Schemes for maternal and child care under a comprehensive program.
**Impact on Investments:**
- Increased public spending on healthcare infrastructure and research will drive growth in healthcare services and technologies.
- Enhanced focus on preventive care and advanced medical infrastructure is expected to improve healthcare accessibility and quality, attracting investments in health tech innovations.
**Stock Recommendations:**
- No specific stock recommendations mentioned.

### Hospitality Sector
**Key Highlights:**
- **Spiritual and Cultural Tourism**: Development of sites like Vishnupath temple, Mahabodhi temple, Kashi Vishwanath corridor, and Nalanda University.
- **GST Simplification**: Efforts to simplify GST for the hospitality sector and comprehensive review of the Income Tax Act.
- **Skill Development**: Initiatives to enhance skills and create job opportunities in the hospitality sector.
**Impact on Investments:**
- Infrastructure development and improved connectivity will benefit the hospitality and tourism sectors by attracting more tourists and boosting local economies.
- Simplification of compliance processes and potential infrastructure status for the hospitality sector could lead to increased investments and economic growth.
**Stock Recommendations:**
- No specific stock recommendations were mentioned.

### Aviation Sector
**Key Highlights:**
- **Airport Revivals**: INR 502 crore allocated to revive 22 airports and start 124 new air routes.
- **Additional Allocations**: INR 1000 crore sanctioned for reviving 50 more airports, helicopters, water aerodromes, and advanced landing grounds.
**Impact on Investments:**
- Enhanced funding for aviation infrastructure is expected to support growth in the aviation sector, benefiting airlines and airport operators.
- Improved connectivity will attract more passengers, boosting revenues for airlines and related businesses.
**Stock Recommendations:**
- No specific stock recommendations mentioned.

### Electric Vehicle (EV) Sector
**Key Highlights:**
- **Incentives and Support**: The budget provides incentives and support to boost the EV sector and related startups.
- **Supportive Ecosystem**: Emphasis on creating a supportive ecosystem for EV infrastructure and production.
**Impact on Investments:**
- Government support will drive growth in EV manufacturing and infrastructure development, encouraging investments in the sector.
- Positive impact on startups and established companies in the EV sector due to enhanced incentives and policy support.
**Stock Recommendations:**
- No specific stock recommendations were mentioned.

# Tax Regime Overview
## New Tax Regime (From FY 2024-25)
### Income Tax Slabs
- **Income up to ₹3,00,000:** No tax
- **Income from ₹3,00,001 to ₹6,00,000:** 5%
- **Income from ₹6,00,001 to ₹9,00,000:** 10%
- **Income from ₹9,00,001 to ₹12,00,000:** 15%
- **Income from ₹12,00,001 to ₹15,00,000:** 20%
- **Income above ₹15,00,000:** 30%
### Deductions and Rebates
- **Standard Deduction:** ₹75,000 (increased from ₹50,000 last year)
- **Rebate eligibility under Section 87A:** Increased to ₹7 lakhs (₹25,000 rebate)
### Surcharge
- **Maximum Surcharge Rate:** Reduced from 37% to 25% \for incomes over ₹5 crore
## Old Tax Regime
### Income Tax Slabs
- **Income up to ₹2,50,000:** No tax
- **Income from ₹2,50,001 to ₹5,00,000:** 5%
- **Income from ₹5,00,001 to ₹10,00,000:** 20%
- **Income above ₹10,00,000:** 30%
### Deductions and Exemptions
- **Standard Deduction:** ₹50,000 
- **Deductions and Exemptions:** Various, including HRA, LTA, 80C (up to ₹1.5 lakhs), 80D, 80E, etc.
## Key Differences
- **Income Tax Slabs:** The new regime offers a higher threshold for no tax (up to
₹3,00,000) compared to the old regime (up to ₹2,50,000). The new regime has more slabs with lower rates up to ₹15,00,000, making it simpler and potentially more beneficial for those with lower deductions.
- **Deductions and Exemptions:** The old regime allows a variety of deductions and exemptions such as HRA, LTA, standard deduction, 80C, 80D, etc. The new regime largely removes these, offering only the standard deduction and a higher rebate eligibility.
- **Surcharge:** The new regime has a reduced maximum surcharge rate of 25% \for
incomes over ₹5 crore, down from 37%.
## Choosing Between the Two Regimes
### Breakeven Threshold
- **For salary income:**
- Income of ₹7,00,000: Benefit in the new regime if deductions are less than ₹1,50,000.
- Income of ₹10,00,000: Benefit in the old regime if deductions exceed ₹2,50,000.
- Income of ₹15,50,000: Benefit in the old regime if deductions exceed ₹3,75,000.
- **For other income:**
- Deductions ≤ ₹1.5 lakhs: New regime beneficial.
- Deductions > ₹3.75 lakhs: Old regime beneficial.
# Tax Regime Overview

## New Tax Regime (From FY 2024-25)

### Income Tax Slabs
- **Income up to ₹3,00,000:** No tax
- **Income from ₹3,00,001 to ₹6,00,000:** 5%
- **Income from ₹6,00,001 to ₹9,00,000:** 10%
- **Income from ₹9,00,001 to ₹12,00,000:** 15%
- **Income from ₹12,00,001 to ₹15,00,000:** 20%
- **Income above ₹15,00,000:** 30%

### Deductions and Rebates
- **Standard Deduction:** ₹75,000 (increased from ₹50,000 last year)
- **Rebate eligibility under Section 87A:** Increased to ₹7 lakhs (₹25,000 rebate)

### Surcharge
- **Maximum Surcharge Rate:** Reduced from 37% to 25% for incomes over ₹5 crore

## Old Tax Regime

### Income Tax Slabs
- **Income up to ₹2,50,000:** No tax
- **Income from ₹2,50,001 to ₹5,00,000:** 5%
- **Income from ₹5,00,001 to ₹10,00,000:** 20%
- **Income above ₹10,00,000:** 30%

### Deductions and Exemptions
- **Standard Deduction:** ₹50,000
- **Deductions and Exemptions:** Various, including HRA, LTA, 80C (up to ₹1.5 lakhs), 80D, 80E, etc.

## Key Differences

- **Income Tax Slabs:** The new regime offers a higher threshold for no tax (up to ₹3,00,000) compared to the old regime (up to ₹2,50,000). The new regime has more slabs with lower rates up to ₹15,00,000, making it simpler and potentially more beneficial for those with lower deductions.
- **Deductions and Exemptions:** The old regime allows a variety of deductions and exemptions such as HRA, LTA, standard deduction, 80C, 80D, etc. The new regime largely removes these, offering only the standard deduction and a higher rebate eligibility.
- **Surcharge:** The new regime has a reduced maximum surcharge rate of 25% for incomes over ₹5 crore, down from 37%.

## Choosing Between the Two Regimes

### Breakeven Threshold
- **For salary income:**
  - Income of ₹7,00,000: Benefit in the new regime if deductions are less than ₹1,50,000.
  - Income of ₹10,00,000: Benefit in the old regime if deductions exceed ₹2,50,000.
  - Income of ₹15,50,000: Benefit in the old regime if deductions exceed ₹3,75,000.

- **For other income:**
  - Deductions ≤ ₹1.5 lakhs: New regime beneficial.
  - Deductions > ₹3.75 lakhs: Old regime beneficial.
  - Deductions between ₹1.5 to ₹3.75 lakhs: Depends on the specific income level.

## Senior Citizens

### Old Regime
- **Senior Citizens (60-80 years):** Exemption up to ₹3,00,000.
- **Super Senior Citizens (above 80 years):** Exemption up to ₹5,00,000.

### New Regime
- Same slabs as general taxpayers, but with higher standard deduction and rebate eligibility.


## Announcements for Professionals using presumptive taxation
###Threshold increase
- The threshold for professionals eligible for presumptive taxation has been increased from ₹50 lakh to ₹75 lakh. This aims to simplify tax compliance for a larger number of professionals by allowing them to declare income at a prescribed rate without maintaining detailed accounts​ (Taxscan)​​ (Tax Guru)​.
### Turnover Limit for Businesses
- For retail businesses under Section 44AD, the turnover limit has been increased from ₹2 crore to ₹3 crore. This change is designed to extend the benefits of the presumptive taxation scheme to more small and medium-sized enterprises​ (Taxscan)​.


## Capital Gains Tax Revisions (Effective from 23rd July 2024)

### Minimum Criteria
- **Criteria for Taxation:** Increased from ₹1,00,000 to ₹1,25,000.

### Asset Types

| Asset Type                  | STCG (Earlier) | STCG (Revised) | Holding Period | LTCG (Earlier) | LTCG (Revised) |
|-----------------------------|----------------|----------------|----------------|----------------|----------------|
| **Listed stocks and equity MFs/ETFs** | 15%            | 20%            | 12 months      | 10%            | 12.50%         |
| **Unlisted shares**         | Slab rate      | Slab rate      | 24 months      | 20% with indexation | 12.50%         |
| **Foreign shares**          | Slab rate      | Slab rate      | 24 months      | 20% with indexation | 12.50%         |
| **Debt MFs and ETFs**       | Slab rate      | Slab rate      | NA             | Slab rate      | Slab rate      |
| **Listed bonds**            | Slab rate      | 20%            | 12 months      | 10%            | 12.50%         |
| **REITs and InVITs**        | 15%            | 20%            | 12 months      | 10%            | 12.50%         |
| **Physical real estate**    | Slab rate      | Slab rate      | 24 months      | 20% with indexation | 12.50%         |
| **Gold/silver ETFs**        | Slab rate      | 20%            | 12 months      | Slab rate      | 12.50%         |
| **Physical gold**           | Slab rate      | Slab rate      | 24 months      | 20% with indexation | 12.50%         |
The recent Budget 2024 announcement introduces significant changes to the tax treatment of ESOP (Employee Stock Ownership Plan) buybacks, impacting startup employees and investors. This is one of the common sources of income for startup employees, and founders.

### Key Highlights:

**New Tax Treatment**:
- Payouts from ESOP buybacks will be treated as dividend income rather than capital gains. - This change subjects employees to potentially higher tax liabilities, as dividend income is
taxed according to the individual's income tax slab rates.

**Previous Tax Regime**:
- Previously, companies paid a 20% tax on share buybacks, and employees faced capital
gains tax when selling shares.
- For listed shares held over one year, the capital gains tax was 10% without indexation
benefits, while unlisted shares held over two years attracted a 20% tax with indexation.

**Effective Date**:
- The new tax rules will be effective from October 1, 2024.
- This change aims to increase government revenue but could reduce the attractiveness of
ESOPs as a compensation tool for startups.

**Employee Impact**:
- Employees exercising their stock options will still pay perquisite tax on the difference
between the market value and the exercise price.
- Upon buyback, the proceeds will now be taxed as dividend income, leading to a higher
cumulative tax burden compared to the previous capital gains tax structure.

**Company Impact**:
- Startups may need to rethink their compensation strategies, considering the increased tax
burden on employees.
- The attractiveness of ESOPs as a tool to attract and retain talent might diminish.

**Market Reaction**:
- The startup ecosystem has expressed concerns, suggesting that the new tax regime could
dampen employee morale and reduce the financial benefits of ESOPs.
- Industry experts are advocating for reconsideration or amendments to balance government
revenue goals with startup growth and employee benefits.

**Comparison with Global Standards**:
- The new policy diverges from global practices where ESOP taxation is typically more
favorable, aiming to encourage employee ownership and participation. 8. **Future Considerations**:
 - There may be further discussions and potential lobbying for policy amendments to mitigate the adverse effects on the startup sector.
- Companies might explore alternative incentive structures to maintain employee motivation and satisfaction.

### Conclusion:
The Budget 2024 proposal to tax ESOP buyback proceeds as dividend income represents a significant shift from the previous capital gains tax treatment, potentially increasing the tax burden for startup employees. While aimed at increasing government revenue, this change could impact the attractiveness of ESOPs and the overall startup ecosystem. Industry stakeholders may seek amendments to ensure a balanced approach that supports both government objectives and the growth of the startup sector.
—--------
The Budget 2024 has introduced a significant change by removing the angel tax, a move aimed at supporting startups and fostering innovation. This information is relevant for startup founders and equity holders of unlisted companies.
Below is a detailed summary of this development:

### Key Highlights:

**Angel Tax Removal**:
- The angel tax, under Section 56(2)(viib) of the Income Tax Act, taxed investments in startups
that were received at a premium over the fair market value.
- Initially introduced in 2012 to curb money laundering through excessive valuations, it had
unintended consequences for genuine startup investments.

**Impact on Startups**:
- The removal of the angel tax is expected to boost the startup ecosystem by eliminating a
significant regulatory hurdle.
- Startups will benefit from increased investments, as investors will no longer face the
additional tax burden on premiums paid above fair market value.

**Investor Sentiment**:
- This change is likely to restore investor confidence, making it easier for startups to attract
funding from angel investors and venture capitalists.
- Enhanced investor sentiment can lead to a more vibrant and dynamic startup landscape,
encouraging innovation and entrepreneurship.

**Regulatory Clarity**:
- The removal provides much-needed regulatory clarity, simplifying the tax environment for
both startups and investors.
- This clarity can lead to a smoother investment process, fostering a more conducive
environment for business growth.

**Government’s Objective**:
- The decision aligns with the government’s broader objective of promoting a startup-friendly
ecosystem and driving economic growth through innovation.
- By removing the angel tax, the government aims to create a more favorable environment for
startups to thrive.
**Broader Economic Impact**:
- The boost in startup activity can have a ripple effect on job creation, technological
advancements, and overall economic development.
- By supporting startups, the government is fostering a culture of innovation that can drive
long-term economic benefits.
**Conclusion**:
- The scrapping of the angel tax in Budget 2024 is a strategic move to support the startup
sector.
- It addresses the concerns of the startup community and investors, paving the way for a
more robust and dynamic startup ecosystem in India.

##Changes in Rental Income taxation.
Applicable to those who receive rental income. Specially affects those running AirBNB type of businesses.
1. Union Budget 2024 mandates rental income from residential properties to be declared under 'income from house property.'
2. This change aims to close tax loopholes and prevent tax evasion by property owners.
3. Rental income can no longer be declared under 'profits and gains of business or
profession.'
4. The new rule ensures only standard deductions are available, limiting misuse of
business expense claims.
5. The changes standardize tax treatment for rental income, impacting tax calculations and
deductions.
6. The new rule requiring rental income from residential properties to be declared under
'income from house property' will be applicable from the financial year 2024-25.

Highlights for Startups and MSMEs Tax Updates:

Angel Tax Abolished: Starting April 1, 2024

Digital Tax on Foreign Companies Abolished: Starting August 2024 Short-Term Capital Gains Tax: Increased to 12.5% from 10% Long-Term Capital Gains Tax: Increased to 20% from 15%

Skilling & Employment:
One-month wage for new formal sector employees Incentives for manufacturing sector employers 1,000 ITIs to be upgraded
Loans for upskilling and higher education

Space Economy:
INR 1,000 Cr venture fund for spacetech startups

Manufacturing Growth:
Reduced customs duty on mobile phones, PCBAs, and chargers to 15% New ecommerce exports hub for MSMEs
Reduced duties on critical minerals and components
Increased customs duty on importing PCBs for telecom

MSME Credit Push:
Credit guarantee for machinery/equipment purchases Enhanced MSME credit assessment by public sector banks
Mudra loan limit doubled to INR 20 Lakh Expansion of TReDS trade invoicing platform New SIDBI branches in major MSME clusters
Digital Business & Infrastructure:
Digital infrastructure push in agriculture, finance, e-commerce, education, and health TDS rate on ecommerce reduced from 1% to 0.1%
(Document - Investments)

##Physical Gold and Sovereign Gold Funds (SGBs)
For those interested in investments in Gold in any format physical or gold bonds.
Customs Duty Reduction: Customs duty on gold cut from 15% to 6%, leading to a 4% price drop.
Increased Demand: Lower gold prices are expected to boost domestic demand.
Impact on SGBs: Returns from Sovereign Gold Bonds might be negatively impacted due to lower gold prices.
Physical Gold Tax Changes: Holding period for LTCG reduced to 24 months; LTCG tax rate is now 12.5%.
Removal of Indexation: Indexation benefits removed for LTCG on gold.
Gold ETFs/Funds Taxation: STCG taxed at 20%, LTCG at 12.5%, making it more tax-efficient. Investment Strategy: Regular investments in gold recommended over market timing. Standardization of Tax Laws: New tax laws bring consistency across asset classes.
Positive Industry Reaction: The industry welcomes the changes but awaits long-term trends. Increased Festive Demand: Potential for gold prices to rise during festive seasons despite current cuts.

```
"""


prompt = PromptTemplate.from_template(template)
parser = StrOutputParser()

chain = prompt | llm | parser


st.sidebar.image("resources/WW_Logo.svg")
st.title("Your Personal Budget Analyser by WiseWealth.me")

# Set up the Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
# creds = ServiceAccountCredentials.from_json_keyfile_name(".streamlit/credentials.json", scope)
creds = st.secrets["GOOGLE_CREDENTIALS"] 
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds, scope)
client = gspread.authorize(creds)
sheet_id = "1fh_RsbhYGcfROSHVjObcbwd20nvLPHu3xN7-rnecYyY"
# Open the Google Sheet
sheet = client.open_by_key(sheet_id).sheet1

def push_to_sheet():
    row = [st.session_state.name, st.session_state.phone,
           ','.join(st.session_state.q1), st.session_state.q2,
           ','.join(st.session_state.q3), ','.join(st.session_state.q4),
           ','.join(st.session_state.q5), st.session_state.q6,
           ','.join(st.session_state.q7)]
    sheet.append_row(row)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1

# Step 1: Form with multi-select and select questions
if st.session_state.step == 1:
    st.subheader("Cut through budget noise with WiseWealth’s free tailored insights.")
    st.write("In this AI age, thousands of smart citizens are cutting through Budget 2024 news with ease!")
    st.write("The FREE AI Assistant from [WiseWealth.me](http://wisewealth.me/) is a groundbreaking way to understand how the budget affects your finances, investments, and taxes. It's fast, simple, and personalized—no two reports are the same.")
    st.write("Just answer a few questions below and get a tailored summary instantly.")

    st.title("Let's get to know you better")

    with st.form(key='survey_form'):
    
        # Define the questions
        q1 = st.multiselect("What of these assets are you invested in or plan to invest in? (Multi-select ones that apply)", [
                        "I don’t invest currently",
                        "Listed stocks and equity MFs/ETFs",
                        "Unlisted shares",
                        "Foreign shares",
                        "Debt MFs and ETFs",
                        "Listed bonds",
                        "REITs and InVITs",
                        "Physical real estate",
                        "Gold/silver ETFs",
                        "Physical Gold",
                        "SGBs(Sovereign Gold Bonds",
                        "Cryptocurrency"
                    ],
                    help="Will help us identify key announcements impacting your investments"
                    )

        q2 = st.selectbox("Do you actively buy and sell assets within a short duration? 1 year for stocks and 2 years for real estate *",  [
                        "Yes, significantly",
                        "Sometimes based on opportunities",
                        "No, I only invest for the long term"
                    ],
                    help="Will help us identify key announcements impacting the taxation of your gains"
                    )

        q3 = st.multiselect("Which of these sectors represents your investment interests?",  [
                        "No Specific Interests",
                        "Infrastructure",
                        "Renewable Energy",
                        "Healthcare",
                        "Agriculture",
                        "Electric Vehicles (EVs)",
                        "Education and Skilling",
                        "Defence",
                        "Telecom",
                        "FMCG - Fast Moving Consumer Goods",
                        "Real Estate and Housing",
                        "Hospitality",
                        "Aviation"
                    ],
                    help="Will help us identify potential long-term upside or downside on your portfolio"
                )

        q4 = st.multiselect("Which of these statements best represents your income sources? (Multi-select that apply) *", [ 
                        "I Receive Salary",
                        "I Recieve Rental Income",
                        "I Run a Small or Medium Size Business (SME) and Earn Profits",
                        "I Run a Startup and Hold Equity",
                        "I Receive or Have Received ESOPs (Employee Stock Options) from my Rmployer",
                        "I am A Professional (Doctor, Lawyer, other Freelancers, etc) who Utilizes Presumptive Taxation (Hint - Select if you fill ITR 4 form while filing ITR)"
                    ],
                    help="Will help us identify key announcements impacting your income taxation"
                )

        q5 = st.multiselect("Do you have any liabilities?",
                    [    
                        "I hold no loans",
                        "Home loan",
                        "Education or skilling loan",
                        "Business Loan",
                        "Other general Loans",
                        
                    ],
                    help="Will help us identify key announcements providing relief or affecting interest payments"
                )
        q6 = st.selectbox("Under what taxation regime do you file your taxes? *",
                    [
                      "I Don't Know My Regime",
                      "New Regime", 
                      "Old Regime"
                    ],
                    help="Will help us understand your compliance with tax regulations"
                )
        q7 = st.multiselect("Do you own or plan to start a business in any of the following sectors",
                    [
                        "No I don't"
                        "Technology",
                        "Retail",
                        "Manufacturing",
                        "Healthcare",
                        "Education",
                        "Finance",
                        "Others"
                    ],
                    help="Will help us identify key business-related opportunities or challenges"
                )

        submit_button = st.form_submit_button("Next")
            
        if submit_button:
            st.session_state.step = 2
            st.session_state.q1 = q1
            st.session_state.q2 = q2
            st.session_state.q3 = q3
            st.session_state.q4 = q4
            st.session_state.q5 = q5
            st.session_state.q6 = q6
            st.session_state.q7 = q7
            st.rerun()

elif st.session_state.step == 2:
    st.title("Just one more step before you get the personalised report!")
    
    name = st.text_input("Name")
    phone = st.text_input("Phone Number")

    if st.button("🔁 Restart"):
        st.session_state.step == 1
        st.rerun()
    
    if st.button("Click Here To Generate Your Personalised Report"):
        st.session_state.step = 3
        st.session_state.name = name
        st.session_state.phone = phone
        st.rerun()

elif st.session_state.step == 3:
    # st.title("Thank You!")
    with st.spinner('⏳ Report generation in progress, this might take a minute. Please don’t refresh...'):
        response = chain.invoke({
            "name" : st.session_state.name,
            "income_source" : ','.join(st.session_state.q4),
            "investment_interest_assets" : ','.join(st.session_state.q1),
            "investment_duration" : st.session_state.q2,
            "invested_sectors" : ','.join(st.session_state.q3),
            "liability_types": ','.join(st.session_state.q5),
            "tax_regime": (st.session_state.q6), 
            "business_sectors": ','.join(st.session_state.q7)
        })
        st.write(response)
        
        if st.button("🔁 Restart"):
            st.session_state.step == 1
            st.rerun()
        push_to_sheet()
