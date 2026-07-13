#qwen modelini mlx-lm kutubxonasi yordamida MacBook M4 Pro xotirasida ishlatish uchun quyidagi kodni yozdik.


import sys
from mlx_lm import load, generate

print("🔄 MacBook M4 Pro xotirasiga Qwen 2.5 modeli yuklanmoqda...")
print("Bu jarayon bir oz vaqt olishi mumkin...\n")

# 1. Model ID va uni yuklash
MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
model, tokenizer = load(MODEL_ID)

print("\n🚀 Model tayyor! Suhbatni boshlashingiz mumkin.")
print("Chiqish uchun 'exit' yoki 'quit' deb yozing.\n")

# 2. Suhbat tarixini (kontekstni) saqlash uchun ro'yxat
# Tizimga model o'zini qanday tutishi kerakligini tushuntiramiz (System Prompt)
messages = [
    {"role": "system", "content": "Siz aqlli, samimiy va foydali sun'iy intellekt yordamchisiz. Savollarga qisqa, aniq va o'zbek tilida javob berasiz."}
]

# 3. Cheksiz suhbat sikli (Chat Loop)
while True:
    try:
        user_input = input("👤 Siz: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Chat yakunlandi. Sog' bo'ling!")
            break
            
        if not user_input.strip():
            continue

        # Foydalanuvchi gapini tarixga qo'shamiz
        messages.append({"role": "user", "content": user_input})

        # MLX va Qwen tushunadigan formatga o'giramiz (Chat Template)
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        print("🤖 Assistant: ", end="", flush=True)

        # Matn generatsiya qilish va ekranga oqim (stream) shaklida chiqarish
        # MLX ichidagi KV Cache shu yerda fonda avtomatik ishlaydi!
        response = generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=512,
            verbose=True 
        )
        print(response)
        print("-" * 40)

        # Modelning javobini ham tarixga qo'shamiz, keyingi savolda asqotadi
        messages.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        print("\nDastur to'xtatildi.")
        break