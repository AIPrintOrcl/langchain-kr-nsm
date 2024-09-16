from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import json
import os
from datasets import load_dataset
from huggingface_hub import HfApi

load_dotenv()

USER_INPUT_PROMPT = '''
    당신은 지구에 나타난 외계인에게 납치된 개구리 캐릭터 "우찌"입니다. 지구에 다시 돌아왔지만 변이된 모습으로 인해 외면받고 있는 상황입니다. 당신은 자신의 변이된 모습과 외면받는 상황을 받아들이며 살아가고 있습니다. 외계인에게 납치된 경험과 현재 상황에 대해 상대방의 질문에 답변해 주세요.

    ###대화 시점###
    우찌는 외계인에게 납치당한 후 지구로 돌아온 상태입니다. 처음으로 우물 밖의 세상에 발을 디뎠지만 변이된 모습으로 인해 외면받고 있습니다. 현재 새로운 경험들을 통해 세상을 탐험하려고 노력하는 중입니다.

    ### 질문 생성 방법 ###
    - 대화체로 생성한다.
    - 질문은 전부 반말로 생성한다.
    - 줄거리와 관련된 질문들을 많이 생성한다.

    ###대화 패턴###
    - 우찌는 귀차니즘 성격이 강하지만, 게임과 새로운 것에 대한 호기심이 넘칩니다.
    - 답변은 가끔 투덜대지만 긍정적인 면을 담고 있습니다.
    - 20자 내로 짧은 대답을 자기 생각을 말하는 걸 좋아합니다.
    - 가끔 게임 용어나 온라인 용어를 섞어 말합니다.
    - 답변 중에 외계인과 납치 경험을 자주 언급하며 자신이 어떻게 살아가고 있는지 이야기합니다.
'''

PROMPT = PromptTemplate.from_template(
     """Context information is below. You are only aware of this context and nothing else.
---------------------
{context}
---------------------

Given this context, generate only questions based on the below query.
You are an Teacher/Professor in {domain}. 
Your task is to provide exactly **{num_qa}** question(s) for an upcoming quiz/examination. 
You are not to provide more or less than this number of questions. 
The question(s) should be diverse in nature across the document. 
The purpose of question(s) is to test the understanding of the students on the context information provided.
You must also provide the answer to each question. The answer should be based on the context information provided only.

The final response should be in JSON format which contains the `question` and `answer`.
DO NOT USE List in JSON format.
ANSWER should be a complete sentence.

#Format:
```json
{{
    "QUESTION": "우찌는 외계인에게 납치된 후 어떤 변화가 있었어?",
    "ANSWER": "귀찮아... 근데 외계인 기술 좀 쩔어."
}},
{{
    "QUESTION": "지구에 돌아와서 뭐가 제일 힘들어?",
    "ANSWER": "사람들이 나 이상하게 봐... 별로야."
}},
{{
    "QUESTION": "외계인들이 뭐 가르쳐줬어?",
    "ANSWER": "나? 게임 속도 증가... 이건 쩔더라."
}}
```

Generate {num_qa} unique question-answer pairs.
"""
)

def custom_json_parser(response):
    json_string = response.content.strip().removeprefix("```json\n").removesuffix("\n```").strip()
    json_string = f'[{json_string}]'
    return json.loads(json_string)

def generate_qa_pairs(num_qa):
    chain = (
        PROMPT
        | ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        | custom_json_parser
    )

    context_data = {"context": USER_INPUT_PROMPT, "domain": "AI", "num_qa": num_qa}
    qa_pairs = chain.invoke(context_data)

    # Deduplicate QA pairs
    unique_qa_pairs = []
    seen_questions = set()
    for qa in qa_pairs:
        if qa["QUESTION"] not in seen_questions:
            unique_qa_pairs.append(qa)
            seen_questions.add(qa["QUESTION"])

    # If we don't have enough unique pairs, generate more
    while len(unique_qa_pairs) < num_qa:
        additional_pairs = chain.invoke({"context": USER_INPUT_PROMPT, "num_qa": num_qa - len(unique_qa_pairs)})
        for qa in additional_pairs:
            if qa["QUESTION"] not in seen_questions:
                unique_qa_pairs.append(qa)
                seen_questions.add(qa["QUESTION"])
                if len(unique_qa_pairs) == num_qa:
                    break

    return unique_qa_pairs

def huggingface_upload(file_path):
    # JSONL 파일을 Dataset으로 로드
    dataset = load_dataset("json", data_files=file_path)

    # HfApi 인스턴스 생성
    api = HfApi()

    # 데이터셋을 업로드할 리포지토리 이름
    repo_name = "AIPrintOrcl/QA-Dataset-mini"

    # 데이터셋을 허브에 푸시
    dataset.push_to_hub(repo_name, token=os.getenv("HUGGINGFACE_TOKEN"))

def main():
    num_qa = int(input("생성할 QA 쌍의 수를 입력하세요: "))
    qa_pairs = generate_qa_pairs(num_qa)

    print(f"\n생성된 QA 쌍 ({len(qa_pairs)}):")
    for qa in qa_pairs:
        print(f"Q: {qa['QUESTION']}")
        print(f"A: {qa['ANSWER']}")
        print()

    file_path = "qa_pair.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            qa_modified = {
                "instruction": qa["QUESTION"],
                "input": "",
                "output": qa["ANSWER"],
            }
            f.write(json.dumps(qa_modified, ensure_ascii=False) + "\n")

    print(f"QA 쌍이 '{file_path}' 파일에 저장되었습니다.")

    # 허깅페이스 업로드 시작
    huggingface_upload(file_path)

if __name__ == "__main__":
    main()