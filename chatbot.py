import requests

API_KEY = "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"  # вставь сюда свой ключ
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mistral-7b-instruct"  # можно выбрать другую модель, см. ниже

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

messages = []

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        break

    messages.append({"role": "user", "content": user_input})

    payload = {
        "model": MODEL,
        "messages": messages
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        data = response.json()

        if "choices" in data:
            reply = data["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": reply})
            print("Bot:", reply)
        else:
            print("Ошибка:", data)

    except Exception as e:
        print("Ошибка при запросе:", str(e))
