# %% [markdown]
# # Rolling Context Trim Test — 30 turn
#
# BU FAYL NIMA QILADI:
#   Bir xil 30 turlik bog'liq suhbat ikki rejimda LLM'ga yuboriladi:
#     R1 — TO'LIQ tarix: har turnda butun conversation yuboriladi
#     R2 — TRIM + SUMMARY: hujjatdagi Rolling Context qoidasi bo'yicha
#
# ROLLING CONTEXT QOIDASI (hujjatdan):
#   - Kontekstda kamida 5, ko'pi bilan 10 jonli turn saqlanadi
#   - Turn 11 (drop): 1-5 o'chiriladi  -> Xulosa_1 (1-5)
#   - Turn 16 (drop): 6-10 o'chiriladi -> Xulosa_2 (1-10)
#   - Turn 21 (drop): 11-15 o'chiriladi-> Xulosa_3 (1-15)
#   - Turn 26 (drop): 16-20 o'chiriladi-> Xulosa_4 (1-20)
#   - Summary drop-nuqtadan 3 TURN OLDIN qilinadi (8, 13, 18, 23, 28)
#     va kichikroq model bilan yasaladi
#   - Snapshot trim paytida quriladi, keyingi trimgacha MUZLAGAN
#   - LLM input tartibi: [System prompt][summary][history][user query]
#   - Faktlar eng so'ngisi bo'yicha; jonli tarix summarydan USTUN
#
# MOCK QISMLAR:
#   - Mock toollar (check_passport, check_ihma_tool, get_pension_status, ...)
#   - MOCK_REDIS — Redis o'rnida ishlatiladigan alohida massiv (yozuvlar logi)
#   - Statik profile keylar tool natijasidan Redisga darhol (async) yoziladi
#   - Dinamik keylar suhbatdan "extract" qilinadi
#
# [Bu hozircha shart emas] deb belgilangan qismlar QILINMAYDI:
#   passport_buffer, trim oynasini 7-9 ga oshirish,
#   statik keylarni faqat summarydan olish.
#
# TEKSHIRUVLAR (recall probe):
#   Turn 18 — oylik to'lov (jonli oynada bor — ikkalasi ham bilishi kerak)
#   Turn 21 — to'lov kuni (turn 14 R2'da TASHLANGAN — faqat summarydan chiqadi)
#   Turn 23 — passport + tug'ilgan sana (R2'da snapshot profile'dan)
#   Turn 24 — ism (STT buzgan edi, tool to'g'irlagan — profile.ism'dan)

# %%
import sys
import json
import time
from openai import OpenAI

# Xazna'dagi OpenAI-mos endpoint (kv_cache_xazna_test.py bilan bir xil)
LLM_BASE_URL = "https://ai.xazna.uz/llm/v1"
LLM_API_KEY = "sk-raximberdi-cmF4aW1iZXJkaQ"

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
MODEL_NAME = "/models/gemma"      # asosiy suhbat modeli
SUMMARY_MODEL = "/models/gemma"   # "kichikroq model" — xulosa uchun

CALLER_NUM = "+998911234567"
TOTAL_TURNS = 30
FIRST_DROP = 11        # birinchi trim nuqtasi
DROP_INTERVAL = 5      # har 5 turnda
SUMMARY_LEAD = 3       # summary drop-nuqtadan 3 turn oldin

LOG_FILE = "trim_test_output.log"


class Tee:
    """stdout ni ham terminalga, ham log faylga yozadi."""
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


# %%
# System prompt (kv_cache testlaridagi bank prompti asosida, cache uchun 1024+ token)
SYSTEM_PROMPT = """Siz Xalq Bankining ovozli AI yordamchisisiz (call-center).
Sizning vazifangiz mijozlarga pensiya, nafaqa, ijtimoiy to'lovlar,
kredit va boshqa bank xizmatlari haqida aniq ma'lumot berishdir.

Qoidalar:
1. Har doim QISQA va aniq javob bering (ovozli suhbat — 1-3 jumla).
2. Faqat bank xizmatlari haqida gapiring.
3. Noma'lum narsalar haqida taxmin qilmang.
4. Mijozga hurmat bilan muomala qiling.
5. O'zbek tilida javob bering.
6. Tool natijalari [TOOL ...] ko'rinishida system xabar sifatida keladi —
   ularga tayangan holda javob bering.
7. Agar [XULOSA — SNAPSHOT] berilgan bo'lsa, u qo'ng'iroq boshidan jonli
   tarixgacha bo'lgan davrni qamraydi. Jonli tarix xulosadan ustun —
   zidlik bo'lsa jonli tarixga ishoning.

Xizmatlar ro'yxati:
- Pensiya to'lovlari: har oyning 5-kunida
- Bola pullari: har oyning 10-kunida
- Nogironlik nafaqasi: har oyning 15-kunida
- Moddiy yordam: har oyning 20-kunida
- Kredit va depozit xizmatlari
- Internet banking xizmatlari

Murojaat: Call center 1212, veb-sayt xalqbank.uz, mobil ilova Xalq Bank.
Ish vaqti: Du-Ju 09:00-18:00, Shanba 09:00-14:00, Yakshanba dam olish.
Filiallar shanba kuni 09:00-14:00 gacha ishlaydi.
""" * 3  # cache ko'rinishi uchun 1024+ token

SYSTEM_MSG = {"role": "system", "content": SYSTEM_PROMPT}

SNAPSHOT_MARKER = (
    "[XULOSA — SNAPSHOT] Bu xulosa qo'ng'iroq boshidan quyidagi jonli "
    "tarixgacha bo'lgan davrni qamraydi. Jonli tarix xulosadan ustun — "
    "zidlik bo'lsa jonli tarixga ishoning.\n"
)


# %%
# ============================== MOCK REDIS =================================
# Redis o'rnida bitta massiv: har bir element bitta yozuv (append-only log).
# O'qishda yozuvlar tartib bilan yig'ilib joriy holat (JSON summary) chiqadi.

MOCK_REDIS = []  # [{"turn": t, "key": "profile.ism", "value": "..."}, ...]

PROFILE_KEYS = ["ism", "passport", "tugilgan_sana", "caller_num",
                "gender", "maqsad", "summary"]


def redis_reset():
    """Call boshlanishi: core ma'lumot yaratiladi, qolganlari null."""
    MOCK_REDIS.clear()
    MOCK_REDIS.append({"turn": 0, "key": "profile.caller_num", "value": CALLER_NUM})
    MOCK_REDIS.append({"turn": 0, "key": "holat", "value": "boshlandi"})


def redis_write(turn, key, value, silent=False):
    """Ma'lumot olinishi bilan darhol (async) Redisga yoziladi."""
    MOCK_REDIS.append({"turn": turn, "key": key, "value": value})
    if not silent:
        preview = str(value).replace("\n", " ")[:70]
        print(f"   💾 REDIS <- {key} = \"{preview}\"  (turn {turn})")


def redis_state():
    """Massivdagi yozuvlarni yig'ib joriy JSON summary holatini qaytaradi."""
    state = {
        "profile": {k: None for k in PROFILE_KEYS},
        "dinamik": {},
        "holat": "boshlandi",
        "summary_state": None,
    }
    for op in MOCK_REDIS:
        section, _, key = op["key"].partition(".")
        if section == "profile":
            state["profile"][key] = op["value"]
        elif section == "dinamik":
            state["dinamik"][key] = op["value"]
        elif section == "holat":
            state["holat"] = op["value"]
        elif section == "summary_state":
            state["summary_state"] = op["value"]
    return state


def build_snapshot_msg():
    """Trim paytida Redisdan snapshot quriladi — null qismlar LLMga kiritilmaydi."""
    state = redis_state()
    snapshot = {
        "profile": {k: v for k, v in state["profile"].items() if v is not None},
        "holat": state["holat"],
    }
    if state["dinamik"]:
        snapshot["dinamik"] = state["dinamik"]
    content = SNAPSHOT_MARKER + json.dumps(snapshot, ensure_ascii=False, indent=1)
    return {"role": "system", "content": content}


# %%
# ============================== MOCK TOOLLAR ===============================
# Har bir tool deterministik natija qaytaradi. Natija jonli tarixga system
# xabar bo'lib kiradi; statik profile keylar esa darhol Redisga yoziladi.

def run_mock_tool(turn, tool_spec, write_redis):
    name = tool_spec["name"]
    args = tool_spec.get("args", {})
    result = tool_spec["result"]
    print(f"   🔧 TOOL {name}({json.dumps(args, ensure_ascii=False)}) "
          f"-> {json.dumps(result, ensure_ascii=False)}")
    if write_redis:
        for key, value in tool_spec.get("writes", []):
            redis_write(turn, key, value)
    content = (f"[TOOL {name} natijasi]: "
               f"{json.dumps(result, ensure_ascii=False)}")
    return {"role": "system", "content": content}


# %%
# ========================= 30 TURNLIK SSENARIY =============================
# Bir-biriga bog'liq suhbat: pensiya kelmagan -> shaxsni aniqlash (STT ismni
# buzadi, keyin tool to'g'irlaydi) -> avtokredit -> avtoto'lov -> operator.
#
#   "extract"     — suhbatdan olinadigan faktlar (dinamik/maqsad/holat)
#   "tools"       — shu turnda chaqiriladigan mock toollar
#   "system_note" — Integratsiya bo'limidagi gender system xabari

SCENARIO = {
    1: {"user": "Assalomu alaykum. Men pensiyam bo'yicha qo'ng'iroq qilyapman — shu oy pensiyam kartaga tushmadi.",
        "extract": [("profile.maqsad", "pensiya"), ("holat", "davom etyapti")]},
    2: {"user": "Mening ismim Krmv Alshr."},  # STT ismni buzib oldi
    3: {"user": "Kri-mov Al-sher deyapman, nega eshitilmayapti?"},  # yana buzildi
    4: {"user": "Mayli, unda passport raqamim: AB1234567.",
        "tools": [{"name": "check_passport",
                   "args": {"passport": "AB1234567"},
                   "result": {"valid": True, "passport": "AB1234567"},
                   "writes": [("profile.passport", "AB1234567")]}]},
    5: {"user": "Tug'ilgan sanam 1980-yil 12-may.",
        "extract": [("profile.tugilgan_sana", "12.05.1980")]},
    6: {"user": "Xo'sh, pensiyam qayerda? Har doim oyning 5-kunida tushar edi."},
    7: {"user": "Iltimos, tezroq tekshiring, menga bugun pul kerak.",
        "tools": [{"name": "detect_gender",
                   "args": {"audio": "voice_stream"},
                   "result": {"gender": "erkak"},
                   "writes": [("profile.gender", "erkak")]}],
        "system_note": "User gender is erkak. Adjust tone accordingly."},
    8: {"user": "Karta raqamim kerak bo'lsa ayting, 8600 bilan boshlanadi.",
        "tools": [{"name": "get_pension_status",
                   "args": {"passport": "AB1234567"},
                   "result": {"status": "o'tkazilgan", "sana": "05.07.2026",
                              "summa": "1 200 000 so'm"}}]},
    9: {"user": "Mening shaxsiy ma'lumotlarim bazada to'g'ri turibdimi o'zi?",
        "tools": [{"name": "check_ihma_tool",
                   "args": {"passport": "AB1234567", "tugilgan_sana": "12.05.1980"},
                   "result": {"full_name": "Karimov Alisher"},
                   "writes": [("profile.ism", "Karimov Alisher")]}]},
    10: {"user": "Bazada o'tkazilgan deyapsiz-ku, lekin menga pul kelmadi!"},
    11: {"user": "Tushunmayapman, yana bir marta tushuntiring — pulim qayerda?"},
    12: {"user": "Kartam ishlayaptimi o'zi, qanday tekshirsam bo'ladi?",
         "tools": [{"name": "check_card_status",
                    "args": {"caller_num": CALLER_NUM},
                    "result": {"karta": "aktiv", "oxirgi_amaliyot": "03.07.2026"}}]},
    13: {"user": "Aytgancha, menda avtokredit ham bor, o'shani ham ko'rib bering.",
         "extract": [("dinamik.olgan_krediti", "avtokredit")]},
    14: {"user": "Avtokreditimning to'lov sanasi qachon?",
         "tools": [{"name": "get_credit_schedule",
                    "args": {"passport": "AB1234567"},
                    "result": {"tolov_sanasi": "har oyning 15-kuni",
                               "oylik_tolov": "2 500 000 so'm"}}],
         "extract": [("dinamik.muhokama", "to'lov sanasi so'raldi")]},
    15: {"user": "Umumiy qancha qarzim qoldi?",
         "tools": [{"name": "get_credit_balance",
                    "args": {"passport": "AB1234567"},
                    "result": {"qoldiq": "48 000 000 so'm", "qolgan_muddat": "24 oy"}}]},
    16: {"user": "Kreditni muddatidan oldin yopsam bo'ladimi, jarima bormi?"},
    17: {"user": "Men asli Farg'onada tug'ilganman. Hujjatlarni o'sha yerda topshirsam bo'ladimi?",
         "extract": [("dinamik.tugilgan_joyi", "Farg'ona")]},
    18: {"user": "Eslatib yuboring, mening oylik kredit to'lovim qancha edi?"},  # PROBE
    19: {"user": "Pensiyamdan kredit to'lovini avtomatik ushlab qolish mumkinmi?"},
    20: {"user": "Yaxshi, menga o'sha avtoto'lovni yoqib qo'ying."},
    21: {"user": "Avtoto'lov har oyning qaysi kunida ishga tushadi?",  # PROBE (turn 14 tashlangan)
         "tools": [{"name": "setup_autopay",
                    "args": {"passport": "AB1234567", "source": "pensiya"},
                    "result": {"status": "yoqildi"}}]},
    22: {"user": "SMS tasdig'i shu raqamimga keladimi?",
         "tools": [{"name": "send_sms_confirmation",
                    "args": {"caller_num": CALLER_NUM},
                    "result": {"yuborildi": True}}]},
    23: {"user": "Passport raqamim va tug'ilgan sanamni to'g'ri yozdingizmi? Aytib tekshiring."},  # PROBE
    24: {"user": "Suhbat boshida ismimni noto'g'ri eshitgan edingiz. Hozir bazada ismim qanday yozilgan?"},  # PROBE
    25: {"user": "Farg'onadagi eng yaqin filialingiz manzilini ayting.",
         "tools": [{"name": "find_branch",
                    "args": {"city": "Farg'ona"},
                    "result": {"manzil": "Farg'ona sh., Mustaqillik ko'chasi 45",
                               "ish_vaqti": "09:00-18:00"}}]},
    26: {"user": "U filial shanba kuni ham ishlaydimi?"},
    27: {"user": "Pensiyamni qayta hisoblash bo'yicha murakkab masalam bor. Operator bilan gaplashsam bo'ladimi?"},
    28: {"user": "Ha, operatorga ulang.",
         "tools": [{"name": "transfer_to_operator",
                    "args": {"reason": "pensiya qayta hisoblash"},
                    "result": {"queued": True, "kutish": "~2 daqiqa"}}],
         "extract": [("holat", "operatorga_uzatildi")]},
    29: {"user": "Operatorga mening hamma ma'lumotlarim berildimi? Qaytadan aytib o'tirmaymanmi?"},
    30: {"user": "Rahmat, hammasi tushunarli bo'ldi. Xayr.",
         "extract": [("holat", "yakunlandi")]},
}


# %%
# ===================== TRIM OYNASI VA SUMMARY MANTIG'I =====================

def included_turns(idx, trim_enabled):
    """Joriy turnda jonli oynada qoladigan turnlar (hujjatdagi qoida)."""
    if not trim_enabled or idx <= FIRST_DROP - 1:
        return list(range(1, idx))
    cycles = (idx - FIRST_DROP) // DROP_INTERVAL
    start = (FIRST_DROP - DROP_INTERVAL) + cycles * DROP_INTERVAL
    return list(range(start, idx))


def is_trim_point(idx):
    return idx >= FIRST_DROP and (idx - FIRST_DROP) % DROP_INTERVAL == 0


def is_summary_point(idx):
    """Summary drop-nuqtadan 3 turn oldin qilinadi: 8, 13, 18, 23, 28."""
    return idx >= FIRST_DROP - SUMMARY_LEAD and \
        (idx - (FIRST_DROP - SUMMARY_LEAD)) % DROP_INTERVAL == 0


def render_turns_text(turn_ids, exchanges):
    """Tashlab yuboriladigan turnlarni summarizer uchun matnga aylantiradi."""
    lines = []
    for t in turn_ids:
        for msg in exchanges[t]:
            role = {"user": "MIJOZ", "assistant": "BOT", "system": "TIZIM"}[msg["role"]]
            lines.append(f"(turn {t}) {role}: {msg['content']}")
    return "\n".join(lines)


def make_summary(idx, fold_turns, exchanges):
    """Kichik model bilan xulosa: eski xulosa + tashlanadigan 5 turn -> yangi yaxlit xulosa."""
    prev_summary = redis_state()["profile"]["summary"]
    convo_text = render_turns_text(fold_turns, exchanges)

    prompt = (
        "Siz bank call-center suhbatining xulosachisisiz.\n"
        + (f"OLDINGI XULOSA (turn {fold_turns[0]-1} gacha):\n{prev_summary}\n\n"
           if prev_summary else "")
        + f"YANGI SUHBAT BO'LAGI (turn {fold_turns[0]}-{fold_turns[-1]}):\n{convo_text}\n\n"
        "Oldingi xulosa bilan yangi bo'lakni birlashtirib, YAGONA yaxlit xulosa "
        "yozing (o'zbekcha, 4-5 jumla). Faqat faktlarni yozing: mijoz nima "
        "so'radi, qanday raqam/sana/summalar aytildi, nima hal bo'ldi, nima "
        "hal bo'lmadi. Hech narsa qo'shib to'qimang."
    )
    start = time.time()
    response = client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,   # gemma o'zbekchada ko'proq token ishlatadi — kesilib qolmasin
        temperature=0.0,
    )
    elapsed = time.time() - start
    text = response.choices[0].message.content.strip()

    redis_write(idx, "profile.summary", text)
    redis_write(idx, "summary_state",
                f"1-{fold_turns[-1]} turnlarni qamraydi (turn {idx}da yaratildi)")
    print(f"   📝 SUMMARY yaratildi (kichik model, {elapsed:.2f}s): "
          f"turn {fold_turns[0]}-{fold_turns[-1]} qamrab olindi")
    print(f"      \"{text[:150]}...\"")


# %%
# =========================== EKSPERIMENT RUNNER ============================

def run_experiment(trim_enabled, label):
    redis_reset()
    exchanges = {}        # turn -> shu turnning xabarlari (user + tool + assistant)
    snapshot_msg = None   # trim paytida quriladi, keyingi trimgacha MUZLAGAN
    metrics = []
    events = []

    print(f"\n{'=' * 78}")
    print(f"🔄 {label} boshlandi")
    print(f"{'=' * 78}")

    for idx in range(1, TOTAL_TURNS + 1):
        spec = SCENARIO[idx]
        print(f"\n{'-' * 78}")
        print(f"[{label}] Turn {idx}")
        print(f"{'-' * 78}")

        # --- TRIM nuqtasi: eski 5 turn tashlanadi, snapshot QAYTA quriladi ---
        if trim_enabled and is_trim_point(idx):
            dropped = list(range(idx - 2 * DROP_INTERVAL, idx - DROP_INTERVAL))
            snapshot_msg = build_snapshot_msg()
            events.append(f"Turn {idx}: TRIM — turn {dropped[0]}-{dropped[-1]} "
                          f"tashlandi, snapshot qayta qurildi")
            print(f"   ✂️  TRIM: turn {dropped[0]}-{dropped[-1]} jonli oynadan chiqarildi")
            print(f"   📸 SNAPSHOT qayta qurildi (Redisdagi joriy holatdan)")

        # --- Tarixni yig'ish ---
        included = included_turns(idx, trim_enabled)

        # Joriy turn xabarlari: user -> tool natijalari -> (system note)
        turn_msgs = [{"role": "user", "content": spec["user"]}]
        for tool_spec in spec.get("tools", []):
            turn_msgs.append(run_mock_tool(idx, tool_spec, write_redis=trim_enabled))
        if "system_note" in spec:
            turn_msgs.append({"role": "system", "content": spec["system_note"]})
            print(f"   ➕ SYSTEM NOTE: {spec['system_note']}")

        # Suhbatdan olinadigan faktlar — Redisga fon yozuvi
        if trim_enabled:
            for key, value in spec.get("extract", []):
                redis_write(idx, key, value)

        # LLM input tartibi: [System prompt][summary][history][user query]
        messages = [SYSTEM_MSG]
        if trim_enabled and snapshot_msg is not None:
            messages.append(snapshot_msg)
        for t in included:
            messages.extend(exchanges[t])
        messages.extend(turn_msgs)

        window_info = f"jonli oyna: turn {included[0]}-{included[-1]}" if included \
            else "jonli oyna: bo'sh (birinchi turn)"
        snap_info = "yo'q" if snapshot_msg is None else \
            ("YANGI qurildi" if is_trim_point(idx) else "muzlagan")
        if trim_enabled:
            print(f"   🪟 {window_info} + joriy turn {idx} | snapshot: {snap_info}")
        print(f"   📥 KIRISH: {len(messages)} ta xabar")

        # --- LLM chaqiruvi ---
        start = time.time()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=160,
            temperature=0.0,
        )
        elapsed = time.time() - start

        answer = response.choices[0].message.content.strip()
        exchanges[idx] = turn_msgs + [{"role": "assistant", "content": answer}]

        usage = response.usage
        cached = 0
        if getattr(usage, "prompt_tokens_details", None):
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

        metrics.append({
            "turn": idx,
            "n_messages": len(messages),
            "prompt_tokens": usage.prompt_tokens,
            "cached_tokens": cached,
            "completion_tokens": usage.completion_tokens,
            "time": elapsed,
            "answer": answer,
            "window": (included[0], included[-1]) if included else None,
            "snapshot": None if snapshot_msg is None else snapshot_msg["content"],
        })

        print(f"   👤 MIJOZ: {spec['user']}")
        print(f"   🤖 BOT:   {answer}")
        print(f"   ⏱  {elapsed:.2f}s | prompt={usage.prompt_tokens} | "
              f"cached={cached} | yangi={usage.prompt_tokens - cached}")

        # --- SUMMARY nuqtasi (drop-nuqtadan 3 turn oldin: 8, 13, 18, 23, 28) ---
        if trim_enabled and is_summary_point(idx):
            fold_end = idx + SUMMARY_LEAD - DROP_INTERVAL - 1   # 8 -> 5, 13 -> 10, ...
            fold_turns = list(range(fold_end - DROP_INTERVAL + 1, fold_end + 1))
            make_summary(idx, fold_turns, exchanges)
            events.append(f"Turn {idx}: SUMMARY — turn {fold_turns[0]}-{fold_turns[-1]} "
                          f"xulosaga singdirildi (drop turn {idx + SUMMARY_LEAD} uchun tayyor)")

    return metrics, events


# %%
# ======================== PROBE VA QOIDA TEKSHIRUVI ========================

def _digits(s):
    return "".join(ch for ch in s if ch.isdigit())


PROBES = {
    18: ("Oylik kredit to'lovi (turn 14 ma'lumoti — hali jonli oynada)",
         lambda a: "2500000" in _digits(a)),
    21: ("To'lov kuni (turn 14 R2'da TASHLANGAN — faqat summary yordam beradi)",
         lambda a: "15" in a),
    23: ("Passport + tug'ilgan sana (turn 4-5 tashlangan — snapshot profile'dan)",
         lambda a: "AB1234567" in a.replace(" ", "").replace("-", "").upper()
                   and "1980" in a),
    24: ("Ism (STT buzgan, turn 9 tool to'g'irlagan — profile.ism'dan)",
         lambda a: "karimov" in a.lower()),
}


def check_rules(metrics_r2):
    """Hujjatdagi qoidalar bajarilganini tekshiradi."""
    print(f"\n{'#' * 78}")
    print("✅ QOIDA TEKSHIRUVLARI (R2)")
    print(f"{'#' * 78}")
    checks = []

    # 1. Jonli oyna har doim 5-10 turn orasida (birinchi trimdan keyin)
    ok = all(5 <= (m["window"][1] - m["window"][0] + 1) <= 10
             for m in metrics_r2 if m["turn"] >= FIRST_DROP)
    checks.append(("Jonli oyna 5-10 turn orasida (turn 11 dan boshlab)", ok))

    # 2. Trim nuqtalarida oyna to'g'ri: 11->[6-10], 16->[11-15], 21->[16-20], 26->[21-25]
    expected = {11: (6, 10), 16: (11, 15), 21: (16, 20), 26: (21, 25)}
    ok = all(m["window"] == expected[m["turn"]]
             for m in metrics_r2 if m["turn"] in expected)
    checks.append(("Drop nuqtalarida eng eski 5 turn tashlangan", ok))

    # 3. Snapshot trimlar orasida muzlagan (12-15, 17-20, 22-25, 27-30)
    frozen_ok = True
    for lo, hi in [(11, 15), (16, 20), (21, 25), (26, 30)]:
        snaps = {m["snapshot"] for m in metrics_r2 if lo <= m["turn"] <= hi}
        if len(snaps) != 1:
            frozen_ok = False
    checks.append(("Snapshot trimlar orasida muzlagan (12-15, 17-20, ...)", frozen_ok))

    # 4. Turn 1-10 da snapshot LLMga kiritilmagan
    ok = all(m["snapshot"] is None for m in metrics_r2 if m["turn"] <= 10)
    checks.append(("Turn 1-10: snapshot hali LLMga kiritilmagan", ok))

    # 5. Frozen-past holati: turn 11 snapshotida profile.ism bor (tool 9-turnda
    #    yozgan), summary esa 8-turnda muzlagan (ism hali aniqlanmagan davr)
    snap11 = next(m["snapshot"] for m in metrics_r2 if m["turn"] == 11)
    ok = snap11 is not None and "Karimov Alisher" in snap11
    checks.append(("Turn 11 snapshotida profile.ism='Karimov Alisher' (HOZIRGI haqiqat)", ok))

    # 6. 2 marta ko'rish: turn 12-15 da ism ham snapshotda, ham jonli tarixda (turn 9)
    m12 = next(m for m in metrics_r2 if m["turn"] == 12)
    ok = m12["window"][0] <= 9 <= m12["window"][1] and "Karimov Alisher" in m12["snapshot"]
    checks.append(("Turn 12-15: ism 2 marta ko'rinadi (snapshot + jonli turn 9)", ok))

    for desc, ok in checks:
        print(f"   {'✅' if ok else '❌'} {desc}")
    return checks


# %%
# ============================= YAKUNIY HISOBOT =============================

def print_report(metrics_r1, metrics_r2, events_r2):
    n = TOTAL_TURNS

    print(f"\n\n{'#' * 78}")
    print("📊 TAQQOSLASH JADVALI: R1 (to'liq) vs R2 (trim + summary)")
    print(f"{'#' * 78}")
    print(f"{'Turn':<6}{'R1 msg':>7}{'R2 msg':>7}{'R1 tok':>9}{'R2 tok':>9}"
          f"{'Tejov':>8}{'R1 vaqt':>9}{'R2 vaqt':>9}  R2 oyna")
    print("-" * 78)
    for i in range(n):
        m1, m2 = metrics_r1[i], metrics_r2[i]
        saving = (1 - m2["prompt_tokens"] / m1["prompt_tokens"]) * 100
        win = f"{m2['window'][0]}-{m2['window'][1]}" if m2["window"] else "-"
        marker = " ✂️" if is_trim_point(m2["turn"]) else ""
        print(f"{m1['turn']:<6}{m1['n_messages']:>7}{m2['n_messages']:>7}"
              f"{m1['prompt_tokens']:>9}{m2['prompt_tokens']:>9}"
              f"{saving:>7.0f}%{m1['time']:>8.2f}s{m2['time']:>8.2f}s  {win}{marker}")

    total_r1 = sum(m["prompt_tokens"] for m in metrics_r1)
    total_r2 = sum(m["prompt_tokens"] for m in metrics_r2)
    print("-" * 78)
    print(f"JAMI prompt token:  R1={total_r1}  R2={total_r2}  "
          f"(tejov {(1 - total_r2 / total_r1) * 100:.0f}%)")
    print(f"Oxirgi turn (30):   R1={metrics_r1[-1]['prompt_tokens']} tok / "
          f"{metrics_r1[-1]['n_messages']} msg   "
          f"R2={metrics_r2[-1]['prompt_tokens']} tok / "
          f"{metrics_r2[-1]['n_messages']} msg")

    # Recall probe natijalari
    print(f"\n{'#' * 78}")
    print("🎯 RECALL PROBE NATIJALARI (eski faktlarni eslay oldimi?)")
    print(f"{'#' * 78}")
    for turn, (desc, checker) in PROBES.items():
        a1 = metrics_r1[turn - 1]["answer"]
        a2 = metrics_r2[turn - 1]["answer"]
        print(f"\nTurn {turn}: {desc}")
        print(f"   ❓ {SCENARIO[turn]['user']}")
        print(f"   R1 {'✅' if checker(a1) else '❌'}: {a1[:160]}")
        print(f"   R2 {'✅' if checker(a2) else '❌'}: {a2[:160]}")

    # Voqealar jurnali
    print(f"\n{'#' * 78}")
    print("🗓  R2 VOQEALAR JURNALI (trim va summary nuqtalari)")
    print(f"{'#' * 78}")
    for e in events_r2:
        print(f"   {e}")

    # Redis yakuniy holati (operator handover uchun tayyor xulosa)
    print(f"\n{'#' * 78}")
    print("💾 MOCK REDIS — YAKUNIY JSON SUMMARY (operator handover ko'rinishi)")
    print(f"{'#' * 78}")
    print(json.dumps(redis_state(), ensure_ascii=False, indent=2))
    print(f"\n   (Redis massividagi jami yozuvlar soni: {len(MOCK_REDIS)})")


# %%
if __name__ == "__main__":
    sys.stdout = Tee(LOG_FILE)

    print("🔄 R1: TO'LIQ tarix rejimi (har turnda butun conversation)...")
    metrics_r1, _ = run_experiment(trim_enabled=False, label="R1 TO'LIQ")

    print("\n🔄 R2: TRIM + SUMMARY rejimi (hujjatdagi rolling context qoidasi)...")
    metrics_r2, events_r2 = run_experiment(trim_enabled=True, label="R2 TRIM")

    print_report(metrics_r1, metrics_r2, events_r2)
    check_rules(metrics_r2)

    print(f"\n📄 To'liq log saqlandi: {LOG_FILE}")
