from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# FastAPI 앱 초기화
app = FastAPI()

# 템플릿 설정
templates = Jinja2Templates(directory="app/templates")

# 환경 변수 로드
load_dotenv()

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory="app/static"), name="static")

class QueryInput(BaseModel):
    query: str

# 전역 변수로 RAG 체인 선언
rag_chain = None

def initialize_chain():
    """
    RAG 체인 초기화 함수
    """
    global rag_chain

    # 파일 경로
    folder_path = "./data"

    # 1. OpenAI 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # 2. FAISS 데이터베이스 로드
    db = FAISS.load_local(
        folder_path=folder_path,
        embeddings=embeddings,
        index_name='faiss_index',
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 3. OpenAI LLM 초기화
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=1000
    )

    # 4. 프롬프트 템플릿 정의
    prompt_template = """
    당신은 감염병 전문가입니다. 반드시 정확한 답을 해주시며, 동일한 질문에는 같은 대답을 해주세요.

    Instructions:
    - 반드시 "retriver"에 검색된 문서만을 활용하여 대답해주세요. 그 외의 참고자료나 창작된 답변은 하지말아주세요.
    - 만일 적절한 대답을 발견하지 못했을 때, '잘 모르겠습니다.'로 대답해주세요. 
    - 아래의 제공된 #Example Format을 참고하여 Markdown 형식으로 대답해주세요.
    - Include references in the "References" section using the source's URL from the metadata.
    - 제시할 출처가 두개 이상일 경우는 괄호를 사용하지 말고 ","로 구분자를 사용해주세요.
    - 모든 대답은 한국어로 해주세요.

    #Example Format (in Markdown)닌
        (detailed answer to the question)\n\n

        **출처**\n
        - (URL of the source)

    #Context:
    {context}

    #Question:
    {question}

    #Answer (in Markdown):
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    # 5. LLM 체인 생성
    llm_chain = prompt | llm

    # 6. RAG 체인 구성
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain | StrOutputParser()

    print("RAG 체인이 성공적으로 초기화되었습니다!")

initialize_chain()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    메인 페이지 렌더링
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def query_endpoint(query_input: QueryInput):
    """
    사용자의 질문을 받아 RAG 체인을 통해 답변 생성
    """
    try:
        if rag_chain is None:
            raise RuntimeError("RAG 체인이 초기화되지 않았습니다.")
        result = rag_chain.invoke(query_input.query)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)