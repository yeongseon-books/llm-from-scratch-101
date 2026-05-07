# llm-from-scratch-101

`llm-from-scratch-101` 시리즈 예제 코드 저장소입니다.
이 저장소는 `numpy`만 사용한 아주 작은 교육용 구현이며, 재현 가능한 동작을 위해 고정 시드를 사용합니다.

## 안내

- 이 코드는 tiny educational model이며 production 용도가 아닙니다.
- 모든 예제는 오프라인에서 빠르게 실행되도록 작은 차원으로 구성했습니다.

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
python ko/01-tokenizer/episode.py
python en/09-chatbot-wrapper/episode.py
pytest -q
```
