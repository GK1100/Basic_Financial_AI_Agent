import os
from dotenv import load_dotenv

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlesearch import GoogleSearch
from phi.playground import Playground, serve_playground_app


load_dotenv()

# Create the websearch agent
web_search_agent=Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,
)

## A Financial Agent 
finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                        company_news=True)
        ],
    instructions=["Use tables to display the data"],
    show_tools_calls=True,
    markdown=True,
   
)



app = Playground(
    agents=[web_search_agent, finance_agent]).get_app()


# .env file
# GROQ_MODEL_ID=your_model_id
# GROQ_API_KEY=your_api_key
# PHI_API_KEY=your_phidata_api_key

if __name__=="__main__":
    serve_playground_app("playground:app",reload=True)