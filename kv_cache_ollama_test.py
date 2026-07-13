import time
import requests

# ==========================================================================
# BU FAYL NIMA QILADI:
#   Ollama orqali lokal ishlayotgan modelga (qwen2.5:7b) 30 turlik uzun
#   suhbat yuboriladi va NATIJADA modelning KV-kesh (KV cache) mexanizmi
#   javob tezligiga qanchalik ta'sir qilishi o'lchanadi.
#
# NIMANI TEST QILADI:
#   To'rtta rejim solishtiriladi:
#     R1 Cache ON  — suhbatning TO'LIQ tarixi har safar yuboriladi, model
#                    sessiyasi (va uning KV-keshi) turnlar orasida saqlanadi.
#     R1 Cache OFF — xuddi shu to'liq tarix yuboriladi, lekin har turndan
#                    oldin model MAJBURAN xotiradan tushiriladi (keshsiz,
#                    "sovuq" holatda qayta hisoblashga majbur qilinadi).
#     R2 Cache ON  — tarix maxsus qoida bo'yicha QIRQILADI (pastga qarang),
#                    kesh yoqilgan holda.
#     R2 Cache OFF — xuddi shu qirqilgan tarix, lekin har turnda kesh
#                    majburan o'chiriladi (R1 OFF kabi).
#   Solishtirish uchun uchta o'lchov ishlatiladi: (1) haqiqiy vaqt (soniya),
#   (2) Ollama'ning o'zi qaytargan prompt_eval_count (necha token qayta
#   hisoblanganini bildiradi) va (3) shu tokenlar uchun taxminiy KV-kesh
#   RAM sarfi (model arxitekturasidan hisoblab chiqiladi).
#
# QANDAY QILADI:
#   - Har bir turn uchun to'liq xabarlar ro'yxati (yoki uning trim qilingan
#     varianti) Ollama'ning /api/chat endpoint'iga POST qilinadi.
#   - "Cache OFF" real vaqt xatti-harakatini simulyatsiya qilish uchun
#     modelga keep_alive=0 bilan "ping" yuborilib, u xotiradan tushiriladi —
#     shu bilan navbatdagi so'rov haqiqiy sovuq (cold) holatdan boshlanadi.
#   - "Trim" rejimida (R2) yagona qoida bor: kontekstda oldingi turnlar
#     KAMIDA 5 TA, KO'PI BILAN 10 TA saqlanadi. Turn 1-10'da hali 10 tadan
#     oshmagani uchun hech narsa tashlanmaydi. Turn 11'dan boshlab, har 5
#     turnda bir marta (11, 16, 21, 26, ...) eng eski 5 talik blok butunlay
#     kontekstdan chiqarib tashlanadi — bu INCLUDED_TURNS_BY_IDX lug'atida
#     avtomatik hisoblab chiqiladi.
#   - Har turndan keyingi Ollama javobidan prompt_eval_count, eval_count,
#     load_duration va prompt_eval_duration kabi xolis metrikalar yig'iladi.
#   - Oxirida barcha 4 rejim uchun 3 ta jadval chop etiladi: vaqt jadvali,
#     token-son jadvali (+ oxirgi turn uchun KV-kesh RAM taxmini) va
#     load/prefill/eval vaqtlari jadvali.
# ==========================================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"


U1 = "Menga sun'iy intellekt tarixi, uning o'rtaga chiqishi, " \
    "1956-yilgi Dartmut konferensiyasi va bugungi transformer " \
    "arxitekturasi inqilobi haqida juda batafsil ma'lumot ber."
U2 = "Rahmat. Endi Alan Tyuring haqida qisqacha yoz."
U3 = "Yaxshi. SI qishlari nima va ular nega sodir bo'lgan?"
U4 = "Transformer modeli nechanchi yili taklif qilingan?"
U5 = "Ajoyib. Hozirgi barcha suhbatimizni 3 ta so'z bilan xulosa qil."
U6 = "BERT modeli haqida ham qisqacha yoz."
U7 = "Xo'sh, ushbu suhbatimizda nechta model haqida gaplashdik?"
U8 = "Yuqorida aytib o'tgan SI qishlari haqida yana bir bor eslatib o'ting-chi?"
U9 = "GPT modeli haqida ham qisqacha ayting."
U10 = "AlphaGo haqida bilasizmi? Qisqacha ayting."
U11 = "ChatGPT qachon chiqarilgan?"
U12 = "Diffusion modellar nima uchun ishlatiladi?"
U13 = "Reinforcement learning nima?"
U14 = "Computer vision sohasi haqida qisqacha ayting."
U15 = "Xo'sh, hozirgacha nechta mavzuni muhokama qildik?"
U16 = "Sun'iy intellektda etika muammolari qanday bo'lishi mumkin?"
U17 = "LLM so'zi nimani anglatadi?"
U18 = "Tokenizatsiya jarayoni haqida qisqacha tushuntiring."
U19 = "Fine-tuning va pretraining o'rtasidagi farq nimada?"
U20 = "Yuqorida AlphaGo haqida gapirgan edik, u aslida nima haqida edi?"
U21 = "Neyron tarmoq nima?"
U22 = "Backpropagation algoritmi qanday ishlaydi?"
U23 = "GAN (Generative Adversarial Network) haqida qisqacha ayting."
U24 = "Overfitting nima va uni qanday oldini olish mumkin?"
U25 = "Yuqorida ChatGPT haqida gapirgan edik, u qachon chiqqan edi?"
U26 = "Embedding vektor nima?"
U27 = "Attention mexanizmi qanday ishlaydi?"
U28 = "Multi-modal modellar nima?"
U29 = "Yuqorida reinforcement learning haqida gapirgan edik, u nima edi?"
U30 = "Yakunda, ushbu suhbatda nechta AI mavzusini ko'rib chiqdik?"

questions = [
    U1, U2, U3, U4, U5, U6, U7, U8, U9, U10,
    U11, U12, U13, U14, U15, U16, U17, U18, U19, U20,
    U21, U22, U23, U24, U25, U26, U27, U28, U29, U30,
]
system_prompt = {"role": "system", "content": "Siz qisqa javob beruvchi yordamchisiz."}

# Trim rejimida (R2) yagona umumiy qoida bor va u BARCHA turnlarga (1'dan 30'gacha) qo'llanadi:
# kontekstda oldingi turnlar KAMIDA 5 TA, KO'PI BILAN 10 TA saqlanadi.
#   Turn 1-10: hali 10 tadan oshmagani uchun hech narsa tashlanmaydi — oddiy o'sib boradi.
#   Turn 11: endi 11 tani saqlash "ko'pi bilan 10" qoidasini buzadi — shuning uchun aynan
#            shu yerdan tashlash boshlanadi (eng eski 5 talik blok, ya'ni 1-5, olib tashlanadi).
# Bundan keyingi davri (16, 21, 26, ...) xuddi shu mantiq bilan avtomatik davom etadi.
INCLUDED_TURNS_BY_IDX = {}

# Turn 11'dan boshlab: har 5 turnda bitta "eng eski 5 talik blok" butunlay tashlab yuboriladi.
# Drop nuqtalari: 11, 16, 21, 26, ... — o'sha nuqtada oldingi 5 turn olib tashlanadi va faqat
# undan keyingi 5 turn (masalan Turn11'da 6-10) qoladi; keyingi 4 turn davomida (masalan
# 12,13,14,15) hech narsa tashlanmay oddiy o'sib boradi, so'ng navbatdagi drop nuqtasida
# (16) yana eng eski 5 talik blok (6-10) olib tashlanadi va h.k.
# Natijada ushlab turilgan oldingi turnlar soni har safar 5 taga tushib, keyin 9-10 tagacha
# o'sib boradi ("kamida 5, ko'pi bilan 10" qoidasi shu tarzda amalga oshiriladi).
_DROP_INTERVAL = 5
_FIRST_DROP_TURN = 11
for _idx in range(_FIRST_DROP_TURN, len(questions) + 1):
    _cycles_passed = (_idx - _FIRST_DROP_TURN) // _DROP_INTERVAL
    _drop_point = _FIRST_DROP_TURN + _cycles_passed * _DROP_INTERVAL
    _base_start = _drop_point - _DROP_INTERVAL
    INCLUDED_TURNS_BY_IDX[_idx] = list(range(_base_start, _idx))


def get_kv_cache_bytes_per_token(model_name):
    """/api/show orqali modelning arxitektura parametrlarini o'qib, KV-kesh
    1 ta kirish tokeni uchun taxminan qancha RAM (bayt) egallashini hisoblaydi.
    Ollama standart holatda KV-keshni f16 (2 bayt/qiymat) da saqlaydi, shu deb hisoblanadi."""
    response = requests.post("http://localhost:11434/api/show", json={"model": model_name})
    model_info = response.json()["model_info"]

    def find(suffix):
        for key, value in model_info.items():
            if key.endswith(suffix):
                return value
        raise KeyError(f"model_info ichida '{suffix}' bilan tugaydigan kalit topilmadi")

    block_count = find(".block_count")            # transformer qatlamlari soni
    head_count = find(".attention.head_count")     # query head'lari soni
    head_count_kv = find(".attention.head_count_kv")  # key/value head'lari soni (GQA)
    embedding_length = find(".embedding_length")
    head_dim = embedding_length // head_count

    BYTES_PER_VALUE = 2  # f16
    K_AND_V = 2
    return block_count * head_count_kv * head_dim * K_AND_V * BYTES_PER_VALUE


def reset_ollama_model():
    """Ollama modelini xotiradan butunlay tushirib, eski KV keshni majburan tozalaydi."""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "keep_alive": 0,  # javobdan so'ng modelni darhol xotiradan tushiradi
    }
    requests.post(OLLAMA_URL, json=payload)
    time.sleep(2)  # model to'liq tushib ketishi uchun kichik kutish


def run_ollama_experiment(kv_cache_enabled, trim_enabled, label):
    exchanges = {}  # turn_idx -> (user_msg, assistant_msg) — har bir turnni alohida saqlaymiz,
                    # shunda istalgan turnni kontekstdan aniq olib tashlash/qoldirish oson bo'ladi
    history_messages = [system_prompt]  # faqat R1 (trim_enabled=False) uchun oddiy ro'yxat sifatida ishlatiladi
    turn_times = []
    turn_metrics = []

    for idx, user_q in enumerate(questions, 1):
        user_msg = {"role": "user", "content": user_q}

        if trim_enabled:
            # Rejim 2: qaysi oldingi turnlar kiritilishini INCLUDED_TURNS_BY_IDX orqali aniqlaymiz.
            # Kalitda yo'q bo'lsa (Turn 1,2,3) — barcha oldingi turnlar kiritiladi (oddiy o'sish).
            included_turns = INCLUDED_TURNS_BY_IDX.get(idx, list(range(1, idx)))
            history_messages = [system_prompt]
            for t in included_turns:
                history_messages.extend(exchanges[t])
            history_messages.append(user_msg)
        else:
            # Rejim 1: To'liq tarix
            history_messages.append(user_msg)

        # Cache OFF bo'lsa, modelni har turndan oldin xotiradan tushirib,
        # to'liq tarixni chindan ham noldan qayta hisoblashga majbur qilamiz.
        if not kv_cache_enabled:
            reset_ollama_model()

        payload = {
            "model": MODEL_NAME,
            "messages": history_messages,
            "stream": False,
            "options": {
                "num_predict": 32,  # Ollama'ning haqiqiy parametri (max_tokens emas!)
                "temperature": 0.0,
            },
        }

        # 👀 LLM'GA NIMA KIRAYOTGANINI TO'LIQ KO'RSATISH (terminalda limit yo'q — hammasi chop etiladi)
        print("\n" + "=" * 70)
        print(f"[{label}] Turn {idx}")
        print("=" * 70)
        print(f"📥 KIRISH — LLM'ga yuborilayotgan to'liq xabarlar ({len(history_messages)} ta):")
        for m_i, m in enumerate(history_messages, 1):
            print(f"   {m_i}. [{m['role']}] {m['content']}")

        start_time = time.time()
        response = requests.post(OLLAMA_URL, json=payload)
        duration = time.time() - start_time
        turn_times.append(duration)

        if response.status_code == 200:
            response_json = response.json()
            response_text = response_json["message"]["content"]
            assistant_msg = {"role": "assistant", "content": response_text}
            history_messages.append(assistant_msg)
            exchanges[idx] = (user_msg, assistant_msg)
            turn_metrics.append(
                {
                    "prompt_eval_count": response_json.get("prompt_eval_count"),  #input number
                    "prompt_eval_duration_s": response_json.get("prompt_eval_duration", 0) / 1e9, #input proccess vaqti
                    "load_duration_s": response_json.get("load_duration", 0) / 1e9, #modelni yuklashga ketadigan vaqt
                    "eval_count": response_json.get("eval_count"), #output number
                    "eval_duration_s": response_json.get("eval_duration", 0) / 1e9, #output proccess vaqti
                }
            )
        else:
            print(f"❌ Xatolik yuz berdi: {response.text}")
            response_text = "Xato"
            assistant_msg = {"role": "assistant", "content": response_text}
            history_messages.append(assistant_msg)
            exchanges[idx] = (user_msg, assistant_msg)
            turn_metrics.append({})

        # 👀 LLM'DAN NIMA CHIQAYOTGANINI VA QANCHA VAQT KETGANINI KO'RSATISH
        print("\n📤 CHIQISH — LLM javobi (to'liq):")
        print(f"   {response_text}")
        print(f"\n⏱ Ketgan vaqt: {duration:.4f} soniya")

    return turn_times, turn_metrics


def print_final_tables(t_r1_on, t_r1_off, t_r2_on, t_r2_off, m_r1_on, m_r1_off, m_r2_on, m_r2_off):
    print("\n\n" + "#" * 70)
    print("📊 OLLAMA REAL PRODUCTION KV CACHE JADVALI (Soniya hisobida)")
    print("#" * 70)
    header = f"{'Turn':<10}{'R1 ON':<12}{'R1 OFF':<12}{'R2 ON':<12}{'R2 OFF':<12}"
    n_turns = len(t_r1_on)
    print(header)
    for i in range(n_turns):
        print(
            f"Turn {i+1:<5}"
            f"{t_r1_on[i]:<12.4f}"
            f"{t_r1_off[i]:<12.4f}"
            f"{t_r2_on[i]:<12.4f}"
            f"{t_r2_off[i]:<12.4f}"
        )

    print("\n" + "#" * 70)
    print("🔬 ISBOT: prompt_eval_count (kirish sifatida hisoblangan tokenlar soni)")
    print("#" * 70)
    print(header)
    for i in range(n_turns):
        print(
            f"Turn {i+1:<5}"
            f"{m_r1_on[i]['prompt_eval_count']!s:<12}"
            f"{m_r1_off[i]['prompt_eval_count']!s:<12}"
            f"{m_r2_on[i]['prompt_eval_count']!s:<12}"
            f"{m_r2_off[i]['prompt_eval_count']!s:<12}"
        )

    last = n_turns - 1
    bytes_per_token = get_kv_cache_bytes_per_token(MODEL_NAME)
    print(f"\n💾 Turn {n_turns} kirish tokenlari uchun taxminiy KV-kesh RAM sarfi "
          f"({bytes_per_token} bayt/token, f16 deb hisoblangan):")
    for name, m in [("R1 ON", m_r1_on), ("R1 OFF", m_r1_off), ("R2 ON", m_r2_on), ("R2 OFF", m_r2_off)]:
        tokens = m[last]["prompt_eval_count"]
        mb = tokens * bytes_per_token / (1024 * 1024)
        print(f"   {name:<8}: {tokens} token → {mb:.2f} MB")

    print("\n" + "#" * 70)
    print("🔬 IKKINCHI ISBOT: load_duration / prompt_eval_duration / eval_duration (soniyada)")
    print("#" * 70)
    wide_header = f"{'Turn':<10}{'R1 ON':<22}{'R1 OFF':<22}{'R2 ON':<22}{'R2 OFF':<22}"
    print(wide_header)
    for i in range(n_turns):
        def fmt(m):
            return f"{m['load_duration_s']:.3f}/{m['prompt_eval_duration_s']:.3f}/{m['eval_duration_s']:.3f}"

        print(
            f"Turn {i+1:<5}"
            f"{fmt(m_r1_on[i]):<22}"
            f"{fmt(m_r1_off[i]):<22}"
            f"{fmt(m_r2_on[i]):<22}"
            f"{fmt(m_r2_off[i]):<22}"
        )


if __name__ == "__main__":
    print("🔄 Ollama: Rejim 1 (Cache ON) bajarilmoqda...")
    reset_ollama_model()
    t_r1_on, m_r1_on = run_ollama_experiment(kv_cache_enabled=True, trim_enabled=False, label="R1 Cache ON")

    print("🔄 Ollama: Rejim 1 (Cache OFF) bajarilmoqda...")
    reset_ollama_model()
    t_r1_off, m_r1_off = run_ollama_experiment(kv_cache_enabled=False, trim_enabled=False, label="R1 Cache OFF")

    print("🔄 Ollama: Rejim 2 (Trimmed ON, KV Cache ON) bajarilmoqda...")
    reset_ollama_model()
    t_r2_on, m_r2_on = run_ollama_experiment(kv_cache_enabled=True, trim_enabled=True, label="R2 Cache ON (Trim)")

    print("🔄 Ollama: Rejim 2 (Trimmed OFF, KV Cache OFF) bajarilmoqda...")
    reset_ollama_model()
    t_r2_off, m_r2_off = run_ollama_experiment(kv_cache_enabled=False, trim_enabled=True, label="R2 Cache OFF (Trim)")

    print_final_tables(t_r1_on, t_r1_off, t_r2_on, t_r2_off, m_r1_on, m_r1_off, m_r2_on, m_r2_off)
