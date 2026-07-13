import time
import os
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ==========================================================================
# BU FAYL NIMA QILADI:
#   OpenAI API orqali 30 turlik uzun suhbat yuboriladi va KV Cache
#   mexanizmi javob tezligiga qanchalik ta'sir qilishi o'lchanadi.
#
# NIMANI TEST QILADI:
#   To'rtta rejim solishtiriladi:
#     R1 Cache ON  — to'liq tarix, cache yoqiq (system prompt o'zgarmaydi)
#     R1 Cache OFF — to'liq tarix, cache o'chiq (system prompt har safar
#                    tasodifiy belgi qo'shiladi → cache miss majburlangan)
#     R2 Cache ON  — trim qilingan tarix, cache yoqiq
#     R2 Cache OFF — trim qilingan tarix, cache o'chiq
#
# CACHE OFF QANDAY QILINADI:
#   OpenAI da "keep_alive=0" yo'q. Buning o'rniga system prompt oxiriga
#   har safar tasodifiy UUID qo'shiladi → prefix o'zgaradi → cache miss! ✅
#
# TRIM QOIDASI (R2):
#   Turn 1-10:  hech narsa tashlanmaydi
#   Turn 11:    1-5 tashlanadi → 6-10 qoladi
#   Turn 16:    6-10 tashlanadi → 11-15 qoladi
#   ...
#   Har doim: KAMIDA 5, KO'PI BILAN 10 turn
#
# O'LCHOVLAR:
#   1. Haqiqiy vaqt (soniya)
#   2. prompt_tokens — jami kirish tokenlari
#   3. cached_tokens — cache dan olingan tokenlar (OpenAI beradi)
#   4. new_tokens    — haqiqatan qayta hisoblangan tokenlar
# ==========================================================================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-4o-mini"  # ← istalgan modelga o'zgartiring

# Katta system prompt — cache ishlashi uchun 1024+ token bo'lishi kerak
SYSTEM_PROMPT_BASE = """Siz Xalq Bankining AI yordamchisisiz.
Sizning vazifangiz mijozlarga pensiya, nafaqa, ijtimoiy to'lovlar,
bola pullari, moddiy yordam va boshqa bank xizmatlari haqida
to'liq va aniq ma'lumot berishdir.

Qoidalar:
1. Har doim qisqa va aniq javob bering.
2. Faqat bank xizmatlari haqida gapiring.
3. Noma'lum narsalar haqida taxmin qilmang.
4. Mijozni hurmat bilan muomala qiling.
5. O'zbek tilida javob bering.

Xizmatlar ro'yxati:
- Pensiya to'lovlari: har oyning 5-kunida
- Bola pullari: har oyning 10-kunida
- Nogironlik nafaqasi: har oyning 15-kunida
- Moddiy yordam: har oyning 20-kunida
- Baraka karta xizmatlari
- Kredit va depozit xizmatlari
- Internet banking xizmatlari
- Valyuta almashtirish xizmatlari

Murojaat qilish:
- Call center: 1212
- Veb-sayt: xalqbank.uz
- Mobil ilova: Xalq Bank

Ish vaqti:
- Dushanbadan jumagacha: 09:00-18:00
- Shanba: 09:00-14:00
- Yakshanba: dam olish kuni

Muhim eslatmalar:
- Barcha to'lovlar Markaziy bank kursi bo'yicha amalga oshiriladi.
- Kechiktirilgan to'lovlar uchun bank javobgar emas.
- Texnik nosozliklar haqida 1212 ga qo'ng'iroq qiling.
""" * 4  # ← 1024+ token bo'lishi uchun ko'paytirildi


# Savollar
questions = [
    "Menga sun'iy intellekt tarixi, uning o'rtaga chiqishi, 1956-yilgi Dartmut konferensiyasi va bugungi transformer arxitekturasi inqilobi haqida juda batafsil ma'lumot ber.",
    "Rahmat. Endi Alan Tyuring haqida qisqacha yoz.",
    "Yaxshi. SI qishlari nima va ular nega sodir bo'lgan?",
    "Transformer modeli nechanchi yili taklif qilingan?",
    "Ajoyib. Hozirgi barcha suhbatimizni 3 ta so'z bilan xulosa qil.",
    "BERT modeli haqida ham qisqacha yoz.",
    "Xo'sh, ushbu suhbatimizda nechta model haqida gaplashdik?",
    "Yuqorida aytib o'tgan SI qishlari haqida yana bir bor eslatib o'ting-chi?",
    "GPT modeli haqida ham qisqacha ayting.",
    "AlphaGo haqida bilasizmi? Qisqacha ayting.",
    "ChatGPT qachon chiqarilgan?",
    "Diffusion modellar nima uchun ishlatiladi?",
    "Reinforcement learning nima?",
    "Computer vision sohasi haqida qisqacha ayting.",
    "Xo'sh, hozirgacha nechta mavzuni muhokama qildik?",
    "Sun'iy intellektda etika muammolari qanday bo'lishi mumkin?",
    "LLM so'zi nimani anglatadi?",
    "Tokenizatsiya jarayoni haqida qisqacha tushuntiring.",
    "Fine-tuning va pretraining o'rtasidagi farq nimada?",
    "Yuqorida AlphaGo haqida gapirgan edik, u aslida nima haqida edi?",
    "Neyron tarmoq nima?",
    "Backpropagation algoritmi qanday ishlaydi?",
    "GAN (Generative Adversarial Network) haqida qisqacha ayting.",
    "Overfitting nima va uni qanday oldini olish mumkin?",
    "Yuqorida ChatGPT haqida gapirgan edik, u qachon chiqqan edi?",
    "Embedding vektor nima?",
    "Attention mexanizmi qanday ishlaydi?",
    "Multi-modal modellar nima?",
    "Yuqorida reinforcement learning haqida gapirgan edik, u nima edi?",
    "Yakunda, ushbu suhbatda nechta AI mavzusini ko'rib chiqdik?",
]

# Trim qoidasi (Ollama versiyasi bilan bir xil)
INCLUDED_TURNS_BY_IDX = {}
_DROP_INTERVAL = 5
_FIRST_DROP_TURN = 11
for _idx in range(_FIRST_DROP_TURN, len(questions) + 1):
    _cycles_passed = (_idx - _FIRST_DROP_TURN) // _DROP_INTERVAL
    _drop_point = _FIRST_DROP_TURN + _cycles_passed * _DROP_INTERVAL
    _base_start = _drop_point - _DROP_INTERVAL
    INCLUDED_TURNS_BY_IDX[_idx] = list(range(_base_start, _idx))


def get_system_prompt(cache_enabled: bool) -> dict:
    """
    Cache OFF uchun: system prompt oxiriga tasodifiy UUID qo'shiladi.
    Bu prefix ni o'zgartiradi → OpenAI cache miss qilishga majbur bo'ladi.
    Cache ON uchun: system prompt doim bir xil → cache hit! ✅
    """
    if cache_enabled:
        content = SYSTEM_PROMPT_BASE
    else:
        # Har safar o'zgarib turuvchi belgi → cache hech qachon ishlamaydi
        random_suffix = f"\n<!-- cache_bust={random.random()} -->"
        content = SYSTEM_PROMPT_BASE + random_suffix

    return {"role": "system", "content": content}


def run_openai_experiment(cache_enabled: bool, trim_enabled: bool, label: str):
    exchanges = {}
    turn_times = []
    turn_metrics = []

    print(f"\n{'=' * 70}")
    print(f"🔄 {label} boshlandi")
    print(f"{'=' * 70}")

    for idx, user_q in enumerate(questions, 1):
        user_msg = {"role": "user", "content": user_q}

        # System prompt (cache on/off ga qarab)
        system_msg = get_system_prompt(cache_enabled)

        # Tarix tuzish
        if trim_enabled:
            included_turns = INCLUDED_TURNS_BY_IDX.get(idx, list(range(1, idx)))
            messages = [system_msg]
            for t in included_turns:
                messages.extend(exchanges[t])
            messages.append(user_msg)
        else:
            if idx == 1:
                messages = [system_msg, user_msg]
            else:
                # Oldingi messages ni saqlab, yangi system prompt bilan yangilaymiz
                # (cache_off holatda har safar yangi random suffix)
                prev_messages = turn_metrics[idx - 2]["messages"] if idx > 1 else []
                messages = [system_msg] + prev_messages[1:] + [user_msg]

        # Terminal chiqishi
        print(f"\n{'=' * 70}")
        print(f"[{label}] Turn {idx}")
        print(f"{'=' * 70}")
        print(f"📥 KIRISH — {len(messages)} ta xabar:")
        for m_i, m in enumerate(messages, 1):
            content_preview = m['content'][:80].replace('\n', ' ')
            print(f"   {m_i}. [{m['role']}] {content_preview}...")

        # API chaqiruv
        start_time = time.time()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=100,
            temperature=0.0,
        )
        elapsed = time.time() - start_time
        turn_times.append(elapsed)

        # Natija
        response_text = response.choices[0].message.content
        assistant_msg = {"role": "assistant", "content": response_text}
        exchanges[idx] = (user_msg, assistant_msg)

        # Metrikalar
        usage = response.usage
        cached_tokens = 0
        if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
            cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0) or 0

        metrics = {
            "prompt_tokens": usage.prompt_tokens,
            "cached_tokens": cached_tokens,
            "new_tokens": usage.prompt_tokens - cached_tokens,
            "completion_tokens": usage.completion_tokens,
            "messages": messages + [assistant_msg],  # keyingi turn uchun
        }
        turn_metrics.append(metrics)

        print(f"\n📤 CHIQISH:")
        print(f"   {response_text[:200]}")
        print(f"\n⏱  Vaqt:            {elapsed:.4f}s")
        print(f"📊 Prompt tokens:   {usage.prompt_tokens}")
        cache_status = "← CACHE ISHLAYAPTI!" if cached_tokens > 0 else "← cache yoq"
        print(f"✅ Cached tokens:   {cached_tokens}  {cache_status}")
        print(f"🆕 Yangi tokens:    {usage.prompt_tokens - cached_tokens}")

    return turn_times, turn_metrics


def print_final_tables(t_r1_on, t_r1_off, t_r2_on, t_r2_off,
                       m_r1_on, m_r1_off, m_r2_on, m_r2_off):

    n = len(t_r1_on)

    # Jadval 1 — Vaqt
    print("\n\n" + "#" * 90)
    print("📊 VAQT JADVALI (soniya)")
    print("#" * 90)
    print(f"{'Turn':<8}{'R1 ON':^14}{'R1 OFF':^14}{'R2 ON':^14}{'R2 OFF':^14}{'R1: ON vs OFF':^16}")
    print("-" * 90)
    for i in range(n):
        speedup = t_r1_off[i] / t_r1_on[i] if t_r1_on[i] > 0 else 1
        print(
            f"Turn {i+1:<3}"
            f"{t_r1_on[i]:^14.3f}"
            f"{t_r1_off[i]:^14.3f}"
            f"{t_r2_on[i]:^14.3f}"
            f"{t_r2_off[i]:^14.3f}"
            f"{speedup:^16.1f}x"
        )

    # Jadval 2 — Token
    print("\n\n" + "#" * 90)
    print("🔬 TOKEN JADVALI (prompt / cached / yangi)")
    print("#" * 90)
    print(f"{'Turn':<8}{'R1 ON':^22}{'R1 OFF':^22}{'R2 ON':^22}{'R2 OFF':^22}")
    print("-" * 90)

    def fmt_tok(m):
        p = m['prompt_tokens']
        c = m['cached_tokens']
        n_tok = m['new_tokens']
        return f"{p}/{c}/{n_tok}"

    for i in range(n):
        print(
            f"Turn {i+1:<3}"
            f"{fmt_tok(m_r1_on[i]):^22}"
            f"{fmt_tok(m_r1_off[i]):^22}"
            f"{fmt_tok(m_r2_on[i]):^22}"
            f"{fmt_tok(m_r2_off[i]):^22}"
        )

    # Xulosa
    print("\n\n" + "#" * 90)
    print("📈 XULOSA")
    print("#" * 90)

    avg_r1_on = sum(t_r1_on) / n
    avg_r1_off = sum(t_r1_off) / n
    avg_r2_on = sum(t_r2_on) / n
    avg_r2_off = sum(t_r2_off) / n

    print(f"O'rtacha vaqt:")
    print(f"  R1 Cache ON  : {avg_r1_on:.3f}s")
    print(f"  R1 Cache OFF : {avg_r1_off:.3f}s  (R1 ON dan {avg_r1_off/avg_r1_on:.1f}x sekin)")
    print(f"  R2 Cache ON  : {avg_r2_on:.3f}s")
    print(f"  R2 Cache OFF : {avg_r2_off:.3f}s")

    last = n - 1
    print(f"\nOxirgi turn ({n}) token sarfi:")
    for name, m in [("R1 ON", m_r1_on), ("R1 OFF", m_r1_off),
                    ("R2 ON", m_r2_on), ("R2 OFF", m_r2_off)]:
        p = m[last]['prompt_tokens']
        c = m[last]['cached_tokens']
        cache_pct = (c / p * 100) if p > 0 else 0
        print(f"  {name:<10}: {p} prompt, {c} cached ({cache_pct:.0f}%)")


if __name__ == "__main__":
    print("🔄 OpenAI: R1 (To'liq tarix, Cache ON) bajarilmoqda...")
    t_r1_on, m_r1_on = run_openai_experiment(
        cache_enabled=True, trim_enabled=False, label="R1 Cache ON"
    )

    print("\n🔄 OpenAI: R1 (To'liq tarix, Cache OFF) bajarilmoqda...")
    t_r1_off, m_r1_off = run_openai_experiment(
        cache_enabled=False, trim_enabled=False, label="R1 Cache OFF"
    )

    print("\n🔄 OpenAI: R2 (Trim, Cache ON) bajarilmoqda...")
    t_r2_on, m_r2_on = run_openai_experiment(
        cache_enabled=True, trim_enabled=True, label="R2 Cache ON"
    )

    print("\n🔄 OpenAI: R2 (Trim, Cache OFF) bajarilmoqda...")
    t_r2_off, m_r2_off = run_openai_experiment(
        cache_enabled=False, trim_enabled=True, label="R2 Cache OFF"
    )

    print_final_tables(t_r1_on, t_r1_off, t_r2_on, t_r2_off,
                       m_r1_on, m_r1_off, m_r2_on, m_r2_off)