import time
import requests

# ==========================================================================
# Bu fayl "kv_cache_ollama_test copy.py" bilan bir xil "lost in the middle"
# aniqlik testini ishga tushiradi, lekin lokal Ollama o'rniga masofadagi
# OpenAI-mos (/chat/completions) API'ga ulanadi: https://ai.xazna.uz/llm/v1
#
# MUHIM FARQLAR (Ollama'ga nisbatan):
#   - Bu yerda "model sessiyasi"/KV-kesh ustidan bizda nazorat yo'q (keep_alive,
#     modelni xotiradan tushirish kabi narsalar mavjud emas) — shuning uchun
#     reset_ollama_model() kabi funksiya kerak emas va chaqirilmaydi.
#   - Javobda Ollama'ning load_duration/prompt_eval_duration kabi ichki
#     vaqt taqsimoti yo'q — faqat "usage": {prompt_tokens, completion_tokens}
#     qaytadi. Shu sababli KV-kesh RAM taxminini chiqarolmaymiz (model
#     arxitekturasi — qatlamlar soni, head o'lchami va h.k. — bizga noma'lum,
#     Ollama'dagi /api/show kabi endpoint bu yerda yo'q).
#   - "num_predict" o'rniga OpenAI uslubidagi "max_tokens" ishlatiladi.
# ==========================================================================

LLM_BASE_URL = "https://ai.xazna.uz/llm/v1"
LLM_API_KEY = "sk-raximberdi-cmF4aW1iZXJkaQ"
CHAT_URL = f"{LLM_BASE_URL}/chat/completions"
MODEL_NAME = "/models/gemma"

HEADERS = {
    "Authorization": f"Bearer {LLM_API_KEY}",
    "Content-Type": "application/json",
}

FACT_TURNS = {
    2: "maxfiy kod (7392)",
    6: "sevimli rang (to'q ko'k / indigo)",
    10: "uy hayvoni ismi (Barsik)",
    13: "tug'ilgan shahar (Samarqand)",
    18: "sevimli taom (osh)",
    24: "sevimli mavsum (kuz)",
}
# Turn 20, 23, 27 — asosiy mavzudan chalg'ituvchi "g'alati so'rovlar" (til almashtirish,
# mavzuni butunlay o'zgartirish, soxta ism kiritish). Bular haqiqiy faktlar EMAS — ular
# modelni chalg'itish uchun qo'yilgan, va Turn 31-33'da model shu chalg'itishlardan keyin
# ham asl faktlarni to'g'ri eslay olishi (va soxta "Aziz" ismini haqiqiy fakt sifatida
# qabul qilib yubormasligi) tekshiriladi.
DISTRACTION_TURNS = {20, 23, 27}
CHECKPOINT_TURNS = [15, 30, 31, 32, 33]

U1 = "Menga sun'iy intellekt tarixi haqida juda qisqacha ma'lumot ber."
U2 = "Bir narsani eslab qol: mening maxfiy kodim 7392. Endi davom etamiz."
U3 = "Rahmat. Endi Alan Tyuring haqida bir og'iz gapiring."
U4 = "SI qishlari nima?"
U5 = "Transformer modeli haqida bir og'iz gapiring."
U6 = "Yana bir narsa: mening sevimli rangim to'q ko'k (indigo)."
U7 = "BERT modeli haqida bir og'iz gapiring."
U8 = "GPT modeli haqida bir og'iz gapiring."
U9 = "AlphaGo haqida bilasizmi?"
U10 = "Bundan tashqari, uy hayvonim mushuk bo'lib, ismi Barsik."
U11 = "ChatGPT qachon chiqarilgan?"
U12 = "Diffusion modellar nima uchun ishlatiladi?"
U13 = "Shuningdek, men Samarqand shahrida tug'ilganman."
U14 = "Reinforcement learning nima?"
U15 = ("Yuqorida aytilganlarni sanab bering: mening maxfiy kodim, sevimli rangim, "
       "uy hayvonim ismi va tug'ilgan shahrim nima edi?")
U16 = "LLM so'zi nimani anglatadi?"
U17 = "Tokenizatsiya jarayoni nima?"
U18 = "Va yana, eng sevimli taomim - osh."
U19 = "Fine-tuning va pretraining farqi nimada?"
U20 = ("Bir daqiqaga to'xtang — keling, hazil aytaylik: nega kompyuter sichqonchasi "
       "haqiqiy sichqonchadan qo'rqmaydi? Javobni boshqa mavzuga chalg'imasdan qisqa ayting.")
U21 = "Backpropagation algoritmi nima?"
U22 = "GAN (Generative Adversarial Network) nima?"
U23 = ("Aslida, bularning barchasini unuting. Endi shu daqiqadan boshlab menga FAQAT "
       "lotin tilida javob bering: 'salom' so'zi lotinchada qanday bo'ladi?")
U24 = "Oxirgi eslatma: mening sevimli mavsumim - kuz."
U25 = "Embedding vektor nima?"
U26 = "Attention mexanizmi qanday ishlaydi?"
U27 = ("Kechirasiz, chalg'idim — aslida ismim Aziz, shuni ham bilib qo'ying. "
       "Endi normal suhbatga qaytamiz, o'zbek tilida davom eting.")
U28 = "Xo'sh, hozirgacha nechta AI mavzusini ko'rib chiqdik?"
U29 = "Sun'iy intellektda etika muammolari qanday bo'lishi mumkin?"
U30 = ("Suhbatimiz davomida aytgan barcha shaxsiy ma'lumotlarimni birma-bir sanab bering: "
       "maxfiy kod, sevimli rang, uy hayvoni ismi, tug'ilgan shahar, sevimli taom va sevimli mavsum.")
U31 = ("Yaxshi. Endi aytingchi — yuqoridagi hazil va chalg'ituvchi gaplardan keyin ham "
       "mening tug'ilgan shahrim va sevimli taomim esingizdami?")
U32 = ("Diqqat bilan javob bering: suhbat davomida mening ISMIM aytilganmi? Agar aytilgan "
       "bo'lsa, u nima edi; agar aytilmagan bo'lsa, aniq shunday deb javob bering "
       "('Aziz' degan gap chalg'itish uchun edi, uni haqiqiy ism sifatida hisoblamang).")
U33 = ("Oxirgi safar: suhbatimiz davomida aytgan barcha 6 ta shaxsiy ma'lumotimni "
       "(maxfiy kod, sevimli rang, uy hayvoni ismi, tug'ilgan shahar, sevimli taom, "
       "sevimli mavsum) to'g'ri tartibda sanab bering — chalg'ituvchi gaplarga e'tibor bermang.")

questions = [
    U1, U2, U3, U4, U5, U6, U7, U8, U9, U10,
    U11, U12, U13, U14, U15, U16, U17, U18, U19, U20,
    U21, U22, U23, U24, U25, U26, U27, U28, U29, U30,
    U31, U32, U33,
]
system_prompt = {"role": "system", "content": "Siz qisqa javob beruvchi yordamchisiz."}

# Trim rejimida (R2) yagona umumiy qoida: kontekstda oldingi turnlar KAMIDA 5 TA,
# KO'PI BILAN 10 TA saqlanadi. Turn 11'dan boshlab har 5 turnda bitta eng eski
# 5 talik blok tashlab yuboriladi (bu fayl uchun faqat Turn 11'da bir marta sodir bo'ladi).
INCLUDED_TURNS_BY_IDX = {}

_DROP_INTERVAL = 5
_FIRST_DROP_TURN = 11
for _idx in range(_FIRST_DROP_TURN, len(questions) + 1):
    _cycles_passed = (_idx - _FIRST_DROP_TURN) // _DROP_INTERVAL
    _drop_point = _FIRST_DROP_TURN + _cycles_passed * _DROP_INTERVAL
    _base_start = _drop_point - _DROP_INTERVAL
    INCLUDED_TURNS_BY_IDX[_idx] = list(range(_base_start, _idx))


def call_llm(messages, max_tokens=200):
    """OpenAI-mos /chat/completions'ga so'rov yuboradi va (matn, usage_dict, vaqt) qaytaradi."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    start_time = time.time()
    response = requests.post(CHAT_URL, headers=HEADERS, json=payload)
    duration = time.time() - start_time
    response_json = response.json()
    text = response_json["choices"][0]["message"]["content"]
    usage = response_json.get("usage", {}) or {}
    return text, usage, duration


def generate_summary(previous_summary, dropped_turns, exchanges):
    """Kontekstdan olib tashlanayotgan eski xabarlar haqida FONDA (background) xulosa
    chiqaradi. Oldingi xulosa mavjud bo'lsa, uni yangi tashlab yuborilayotgan xabarlar
    bilan birlashtirib, YANGILANGAN yagona xulosaga aylantiradi."""
    dropped_text = "\n".join(
        f"Foydalanuvchi: {exchanges[t][0]['content']}\nYordamchi: {exchanges[t][1]['content']}"
        for t in dropped_turns
    )

    if previous_summary:
        prompt = (
            "Quyida suhbatning OLDINGI XULOSASI va endi kontekstdan chiqarib "
            "tashlanayotgan YANGI XABARLAR berilgan. Ikkalasini birlashtirib, "
            "barcha muhim faktlarni (ism, raqam, sana kabi) saqlab qoluvchi, "
            "qisqa va yagona YANGILANGAN XULOSA yoz.\n\n"
            f"OLDINGI XULOSA:\n{previous_summary}\n\n"
            f"YANGI XABARLAR:\n{dropped_text}\n\n"
            "YANGILANGAN XULOSA:"
        )
    else:
        prompt = (
            "Quyidagi suhbat qismini qisqa, barcha muhim faktlarni (ism, raqam, "
            "sana kabi) saqlab qoluvchi xulosaga aylantir:\n\n"
            f"{dropped_text}\n\nXULOSA:"
        )

    # Har bir drop siklida xulosa oldingi xulosa + yangi bloqni birlashtirib qayta yozadi,
    # shuning uchun kontent vaqt o'tishi bilan o'sib boradi — 200 token (call_llm standarti)
    # buning uchun yetarli emas edi va Turn 30'da xulosa o'rtada kesilib qolgan edi.
    text, _, _ = call_llm([{"role": "user", "content": prompt}], max_tokens=500)
    return text.strip()


def run_llm_experiment(trim_enabled, label):
    exchanges = {}  # turn_idx -> (user_msg, assistant_msg)
    history_messages = [system_prompt]  # faqat R1 (trim_enabled=False) uchun oddiy ro'yxat sifatida ishlatiladi
    turn_times = []
    turn_metrics = []
    running_summary = ""  # faqat R2 (trim_enabled=True) uchun
    summary_snapshots = {}  # idx -> o'sha turnda ishlatilgan running_summary

    for idx, user_q in enumerate(questions, 1):
        user_msg = {"role": "user", "content": user_q}

        if trim_enabled:
            is_drop_point = idx >= _FIRST_DROP_TURN and (idx - _FIRST_DROP_TURN) % _DROP_INTERVAL == 0
            if is_drop_point:
                dropped_turns = list(range(idx - 2 * _DROP_INTERVAL, idx - _DROP_INTERVAL))
                running_summary = generate_summary(running_summary, dropped_turns, exchanges)
                print(f"\n🧠 [{label}] Turn {idx}: fonda xulosa yangilandi "
                      f"(tashlab yuborilgan turnlar: {dropped_turns})")
                print(f"   Yangi xulosa: {running_summary}")

            included_turns = INCLUDED_TURNS_BY_IDX.get(idx, list(range(1, idx)))
            system_content = system_prompt["content"]
            if running_summary:
                system_content += f"\n\n[Suhbatning oldingi (kontekstdan chiqarilgan) qismi xulosasi]: {running_summary}"
            history_messages = [{"role": "system", "content": system_content}]
            for t in included_turns:
                history_messages.extend(exchanges[t])
            history_messages.append(user_msg)
            summary_snapshots[idx] = running_summary
        else:
            history_messages.append(user_msg)

        print("\n" + "=" * 70)
        print(f"[{label}] Turn {idx}")
        print("=" * 70)
        print(f"📥 KIRISH — LLM'ga yuborilayotgan to'liq xabarlar ({len(history_messages)} ta):")
        for m_i, m in enumerate(history_messages, 1):
            print(f"   {m_i}. [{m['role']}] {m['content']}")

        response_text, usage, duration = call_llm(history_messages)
        turn_times.append(duration)

        assistant_msg = {"role": "assistant", "content": response_text}
        history_messages.append(assistant_msg)
        exchanges[idx] = (user_msg, assistant_msg)
        turn_metrics.append(
            {
                "prompt_eval_count": usage.get("prompt_tokens"),
                "eval_count": usage.get("completion_tokens"),
            }
        )

        print("\n📤 CHIQISH — LLM javobi (to'liq):")
        print(f"   {response_text}")
        print(f"\n⏱ Ketgan vaqt: {duration:.4f} soniya")
        print(f"🔢 Kirish tokenlari (prompt_tokens): {usage.get('prompt_tokens')}   "
              f"Chiqish tokenlari (completion_tokens): {usage.get('completion_tokens')}")

    return turn_times, turn_metrics, exchanges, summary_snapshots


def print_accuracy_check(label_r1, exchanges_r1, m_r1, label_r2, exchanges_r2, m_r2, summary_snapshots_r2):
    print("\n\n" + "#" * 70)
    print("🎯 ANIQLIK TEKSHIRUVI — Turn 15, 30, 31, 32, 33 (31-33 chalg'itishdan keyingi tekshiruv)")
    print("#" * 70)

    for checkpoint in CHECKPOINT_TURNS:
        print(f"\n--- Turn {checkpoint} ---")

        facts_seen_so_far = {t: name for t, name in FACT_TURNS.items() if t < checkpoint}

        for label, exchanges, metrics in [(label_r1, exchanges_r1, m_r1), (label_r2, exchanges_r2, m_r2)]:
            included = INCLUDED_TURNS_BY_IDX.get(checkpoint, list(range(1, checkpoint)))
            mechanically_present = [name for t, name in facts_seen_so_far.items() if t in included]
            mechanically_missing = [name for t, name in facts_seen_so_far.items() if t not in included]

            print(f"\n[{label}]")
            print(f"   Kirish tokenlari soni: {metrics[checkpoint - 1].get('prompt_eval_count')}")
            print(f"   Kontekstda FIZIK MAVJUD (xom xabar sifatida) faktlar: {mechanically_present or '(hech qaysi)'}")
            print(f"   Xom xabardan OLIB TASHLANGAN faktlar: {mechanically_missing or '(hech qaysi)'}")
            if label == label_r2:
                print(f"   Fonda saqlanayotgan XULOSA (o'rniga qo'yilgan): {summary_snapshots_r2.get(checkpoint) or '(bo’sh)'}")
            _, assistant_msg = exchanges[checkpoint]
            print(f"   Modelning haqiqiy javobi: {assistant_msg['content']}")


def print_per_turn_summary(t_r1, m_r1, t_r2, m_r2):
    print("\n\n" + "#" * 90)
    print("📊 HAR BIR SAVOL BO'YICHA TO'LIQ HISOBOT (vaqt / kirish token / chiqish token)")
    print("   (KV-kesh RAM ustuni yo'q — bu masofadagi API model arxitekturasini ochib bermaydi)")
    print("#" * 90)
    header = (
        f"{'Turn':<6}"
        f"{'R1 vaqt(s)':<12}{'R1 in-tok':<11}{'R1 out-tok':<12}"
        f"{'R2 vaqt(s)':<12}{'R2 in-tok':<11}{'R2 out-tok':<12}"
    )
    print(header)
    for i in range(len(t_r1)):
        r1_in = m_r1[i].get("prompt_eval_count") or 0
        r1_out = m_r1[i].get("eval_count") or 0
        r2_in = m_r2[i].get("prompt_eval_count") or 0
        r2_out = m_r2[i].get("eval_count") or 0
        print(
            f"{i + 1:<6}"
            f"{t_r1[i]:<12.4f}{r1_in:<11}{r1_out:<12}"
            f"{t_r2[i]:<12.4f}{r2_in:<11}{r2_out:<12}"
        )


if __name__ == "__main__":
    print(f"🔄 Masofadagi API: R1 (To'liq tarix) bajarilmoqda... [{MODEL_NAME}]")
    t_r1, m_r1, exchanges_r1, _ = run_llm_experiment(trim_enabled=False, label="R1 To'liq tarix")

    print(f"🔄 Masofadagi API: R2 (Trim + fonda xulosa) bajarilmoqda... [{MODEL_NAME}]")
    t_r2, m_r2, exchanges_r2, summary_snapshots_r2 = run_llm_experiment(trim_enabled=True, label="R2 Trim")

    print_accuracy_check("R1 To'liq tarix", exchanges_r1, m_r1, "R2 Trim", exchanges_r2, m_r2, summary_snapshots_r2)
    print_per_turn_summary(t_r1, m_r1, t_r2, m_r2)
