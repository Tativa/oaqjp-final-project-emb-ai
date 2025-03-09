import requests
import json
from textblob import TextBlob  # Подключаем TextBlob для анализа

def emotion_detector(text_to_analyse):
    # 🔹 Попробуем сначала получить данные из API Watson
    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    myobj = {"raw_document": {"text": text_to_analyse}}

    try:
        response = requests.post(url, json=myobj, headers=headers)
        response.raise_for_status()  # Проверяем ошибки HTTP (4xx, 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса к API: {e}")
        return analyse_with_textblob(text_to_analyse)  # 🔹 Если ошибка → используем TextBlob

    try:
        parsed_response = response.json()
    except json.JSONDecodeError:
        return analyse_with_textblob(text_to_analyse)  # 🔹 Если ошибка JSON → используем TextBlob

    if "emotionPredictions" not in parsed_response or not parsed_response["emotionPredictions"]:
        return analyse_with_textblob(text_to_analyse)  # 🔹 Если API пустой → используем TextBlob

    emotions = parsed_response["emotionPredictions"][0]

    # 🔹 Оставляем только числа
    valid_emotions = {k: v for k, v in emotions.items() if isinstance(v, (int, float))}

    if not valid_emotions:
        return analyse_with_textblob(text_to_analyse)  # 🔹 Если API не вернул числовые эмоции

    # 🔹 Определяем доминирующую эмоцию
    dominant_emotion = max(valid_emotions, key=valid_emotions.get)

    return {
        "anger": valid_emotions.get("anger", 0),
        "disgust": valid_emotions.get("disgust", 0),
        "fear": valid_emotions.get("fear", 0),
        "joy": valid_emotions.get("joy", 0),
        "sadness": valid_emotions.get("sadness", 0),
        "dominant_emotion": dominant_emotion
    }

def analyse_with_textblob(text):
    """Анализируем эмоции через TextBlob, если API Watson не сработал"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.2:
        return {"anger": 0, "disgust": 0, "fear": 0, "joy": 1.0, "sadness": 0, "dominant_emotion": "joy"}
    elif polarity < -0.2:
        return {"anger": 1.0, "disgust": 0, "fear": 0, "joy": 0, "sadness": 0, "dominant_emotion": "anger"}
    else:
        return {"anger": 0, "disgust": 0, "fear": 0, "joy": 0, "sadness": 1.0, "dominant_emotion": "sadness"}
