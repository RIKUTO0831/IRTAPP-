import math
import sys
import json
import numpy as np

from flask import Flask, redirect, render_template_string, request, url_for
import google.generativeai as genai

# ==========================================
# ⚠️ ここに取得したGeminiのAPIキーを貼り付けてください
# ==========================================
genai.configure(api_key="")

# --- 事前定義データ ---
AXES = ["計画性・誠実性", "協調性", "主導的外向性", "挑戦志向・開放性", "情緒安定性"]

# 画像のリスト
CLIENT_IMAGES = [
    "c (1).png",
    "c (2).png",
    "c (3).png",
    "c (4).png",
    "c (5).png"
]

# Raschモデルにより仮想営業職データ500人分から推定したb値を使用
# aは1.0固定のため、1PL/Raschモデルとして扱う
# この値は実データではなく、ハッカソン用の仮想データに基づく
QUESTIONS = [
    {"id": "Q1", "text": "予定が変わったとき、まず作業の順番を組み直すことが多いですか？", "axis": "計画性・誠実性", "keying": "forward", "a": 1.0, "b": -0.5467},
    {"id": "Q2", "text": "締切がまだ先でも、必要そうな情報を先に集め始めることが多いですか？", "axis": "計画性・誠実性", "keying": "forward", "a": 1.0, "b": -0.0481},
    {"id": "Q3", "text": "やることが増えると、優先順位を決めずに目についたものから始めることが多いですか？", "axis": "計画性・誠実性", "keying": "reverse", "a": 1.0, "b": -0.9232},
    {"id": "Q4", "text": "細部が少し曖昧でも、先に前へ進めたくなることが多いですか？", "axis": "計画性・誠実性", "keying": "reverse", "a": 1.0, "b": 0.0481},
    {"id": "Q5", "text": "意見が違っても、いったん相手の考え方を最後まで聞くことが多いですか？", "axis": "協調性", "keying": "forward", "a": 1.0, "b": -0.4976},
    {"id": "Q6", "text": "議論が長引くくらいなら、自分の案を押し切る方が早いと感じることが多いですか？", "axis": "協調性", "keying": "reverse", "a": 1.0, "b": -0.1440},
    {"id": "Q7", "text": "納得できない指摘を受けると、すぐに態度へ出やすいですか？", "axis": "協調性", "keying": "reverse", "a": 1.0, "b": -0.2119},
    {"id": "Q8", "text": "役割分担でもめそうなとき、全員が受け入れやすい折衷案を探す方ですか？", "axis": "協調性", "keying": "forward", "a": 1.0, "b": -0.0377},
    {"id": "Q9", "text": "初対面が多い場でも、自分から会話を始める方ですか？", "axis": "主導的外向性", "keying": "forward", "a": 1.0, "b": -0.6173},
    {"id": "Q10", "text": "方針が止まっている場面では、進め方を口に出すことが多いですか？", "axis": "主導的外向性", "keying": "forward", "a": 1.0, "b": -0.3083},
    {"id": "Q11", "text": "発表役は、できるだけ他の人に任せたい方ですか？", "axis": "主導的外向性", "keying": "reverse", "a": 1.0, "b": -0.3475},
    {"id": "Q12", "text": "大人数の場では、考えがあっても発言を後回しにしがちですか？", "axis": "主導的外向性", "keying": "reverse", "a": 1.0, "b": -0.2692},
    {"id": "Q13", "text": "うまくいっているやり方があっても、別の方法を一度は試したくなることがありますか？", "axis": "挑戦志向・開放性", "keying": "forward", "a": 1.0, "b": -0.1538},
    {"id": "Q14", "text": "初めて触るツールや技術でも、まず少し触って感触を確かめたい方ですか？", "axis": "挑戦志向・開放性", "keying": "forward", "a": 1.0, "b": -0.5562},
    {"id": "Q15", "text": "慣れたやり方があるなら、新しい方法はあまり試さない方ですか？", "axis": "挑戦志向・開放性", "keying": "reverse", "a": 1.0, "b": -0.5260},
    {"id": "Q16", "text": "制約が多い状況では、奇抜そうでも新しい切り口を考えるのが好きですか？", "axis": "挑戦志向・開放性", "keying": "forward", "a": 1.0, "b": 0.4071},
    {"id": "Q17", "text": "締切が重なっても、何から手をつけるかを比較的落ち着いて考えられますか？", "axis": "情緒安定性", "keying": "forward", "a": 1.0, "b": -0.2505},
    {"id": "Q18", "text": "予定外のトラブルが起きると、そのことが頭から離れにくいですか？", "axis": "情緒安定性", "keying": "reverse", "a": 1.0, "b": -0.2023},
    {"id": "Q19", "text": "厳しい指摘を受けても、必要な部分だけ拾って切り替えやすい方ですか？", "axis": "情緒安定性", "keying": "forward", "a": 1.0, "b": -0.0677},
    {"id": "Q20", "text": "作業が思うように進まないと、手が止まるほど気持ちが乱れやすいですか？", "axis": "情緒安定性", "keying": "reverse", "a": 1.0, "b": 0.2686},
]

# 通信エラー時のフォールバック用質問
FALLBACK_CUSTOM_QUESTIONS = [
    {"id": "C1", "text": "目標達成のためなら、多少の犠牲は仕方ないと思いますか？", "axis": "計画性・誠実性", "keying": "forward"},
    {"id": "C2", "text": "自分の意見を曲げてまで、チームの和を保つことが多いですか？", "axis": "協調性", "keying": "forward"},
    {"id": "C3", "text": "誰もやりたがらない役割でも、必要なら自分から引き受けますか？", "axis": "主導的外向性", "keying": "forward"},
    {"id": "C4", "text": "全く経験のない分野の仕事でも、面白そうなら飛び込みますか？", "axis": "挑戦志向・開放性", "keying": "forward"},
    {"id": "C5", "text": "失敗を引きずらず、翌日には気持ちを切り替えられる方ですか？", "axis": "情緒安定性", "keying": "forward"},
]

IDEAL_PROFILE = { "計画性・誠実性": 1.0, "協調性": 0.5, "主導的外向性": 1.5, "挑戦志向・開放性": 0.8, "情緒安定性": 1.2 }
EPS = 1e-9

# --- 分析ロジック ---
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

def convert_answer_to_u(answer: str, keying: str) -> int:
    if keying == "forward": return 1 if answer == "yes" else 0
    if keying == "reverse": return 0 if answer == "yes" else 1
    return 0

def estimate_axis_theta(axis: str, answers: dict, question_defs: list[dict]) -> dict:
    axis_questions = [q for q in question_defs if q["axis"] == axis and q["id"] in answers]
    if not axis_questions:
        return {"axis": axis, "theta": 0.0, "level": "中間"}
        
    theta_grid = np.linspace(-4, 4, 1601)
    u = np.array([convert_answer_to_u(answers[q["id"]], q["keying"]) for q in axis_questions], dtype=float)
    a = np.array([q["a"] for q in axis_questions], dtype=float)
    b = np.array([q["b"] for q in axis_questions], dtype=float)

    p = np.clip(sigmoid(a * (theta_grid[:, None] - b)), EPS, 1.0 - EPS)
    log_likelihood = np.sum(u * np.log(p) + (1.0 - u) * np.log(1.0 - p), axis=1)
    log_posterior = log_likelihood - (theta_grid**2) / 2.0
    theta = float(theta_grid[int(np.argmax(log_posterior))])
    
    level = "高め" if theta >= 0.5 else "低め" if theta <= -0.5 else "中間"
    return {"axis": axis, "theta": round(theta, 3), "level": level}

def estimate_thetas(answers: dict, question_defs: list[dict]) -> list[dict]:
    return [estimate_axis_theta(axis, answers, question_defs) for axis in AXES]

def calculate_sales_aptitude(estimates: list[dict]) -> int:
    total_diff = sum(abs(est["theta"] - IDEAL_PROFILE.get(est["axis"], 0.0)) for est in estimates)
    return max(0, min(100, int(100 - (total_diff * 5))))

# --- Gemini API (独自質問の生成) ---
def generate_custom_questions(estimates: list[dict]) -> list[dict]:
    est_str = ", ".join([f"{e['axis']}:{e['level']}" for e in estimates])
    prompt = f"""適性検査の深掘り質問を各軸1問ずつ（計5問）生成してください。
【ユーザー特性】{est_str} (このレベルに合わせ、判断に迷う絶妙な質問にすること)
【出力条件】
- 「はい」「いいえ」で答えられる1文のみ。
- 高速化のため、各質問文は40文字以内。
- 必ず以下のJSONフォーマットのみ出力。
{{
  "custom_questions": [
    {{"id": "C1", "axis": "計画性・誠実性", "text": "【質問文】", "keying": "forward"}},
    {{"id": "C2", "axis": "協調性", "text": "【質問文】", "keying": "forward"}},
    {{"id": "C3", "axis": "主導的外向性", "text": "【質問文】", "keying": "forward"}},
    {{"id": "C4", "axis": "挑戦志向・開放性", "text": "【質問文】", "keying": "forward"}},
    {{"id": "C5", "axis": "情緒安定性", "text": "【質問文】", "keying": "forward"}}
  ]
}}"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        res = model.generate_content(prompt, generation_config=genai.GenerationConfig(response_mime_type="application/json"))
        data = json.loads(res.text)
        qs = data.get("custom_questions", FALLBACK_CUSTOM_QUESTIONS)
        
        for q in qs:
            q["a"] = 1.0
            axis_theta = next((e["theta"] for e in estimates if e["axis"] == q["axis"]), 0.0)
            q["b"] = axis_theta
        return qs
    except Exception as e:
        for q in FALLBACK_CUSTOM_QUESTIONS:
            q["a"] = 1.0
            axis_theta = next((e["theta"] for e in estimates if e["axis"] == q["axis"]), 0.0)
            q["b"] = axis_theta
        return FALLBACK_CUSTOM_QUESTIONS

# --- Gemini API (AIロープレ1問ずつ動的生成) ---
def generate_roleplay_step(industry: str, estimates: list[dict], step_index: int, client_data: dict, history: list) -> dict:
    phases = [
        {"phase": "1. 初回訪問", "axis": "顧客理解"},
        {"phase": "2. 課題ヒアリング", "axis": "課題発見"},
        {"phase": "3. 解決策の提示", "axis": "提案力"},
        {"phase": "4. 懸念への対応", "axis": "信頼形成"},
        {"phase": "5. トラブル対応", "axis": "ストレス耐性"}
    ]
    current = phases[step_index]
    est_str = ", ".join([f"{e['axis']}:{e['level']}" for e in estimates])

    if step_index == 0:
        prompt = f"""あなたは営業シナリオ作成AIです。
業界: {industry}
ユーザー特性: {est_str} (特性に合わせて顧客の要求難易度を調整)
現在のフェーズ: {current['phase']} (評価軸: {current['axis']})

架空の取引先との最初の商談状況と、トレードオフとなる2択の行動選択肢を生成してください。
【文字数制限】状況の解像度を上げるため、situationは120〜150文字程度で情景や顧客の意図が伝わるように描写し、option_a/bはそれぞれ60文字程度で記述すること。
【出力形式】必ずJSONのみ。
{{
  "client_name": "株式会社〇〇 担当✕✕",
  "client_context": "【背景・課題を60文字程度で】",
  "phase": "{current['phase']}",
  "axis": "{current['axis']}",
  "situation": "【商談相手の最初の発言や状況の詳細な描写】",
  "question": "どう対応しますか？",
  "option_a": "【行動A】",
  "option_b": "【行動B】"
}}"""
    else:
        recent_history = history[-2:] if len(history) > 2 else history
        history_str = "\n".join([f"- フェーズ「{h['phase']}」: 顧客「{h['situation']}」→ ユーザー「{h['chosen_option']}」" for h in recent_history])

        prompt = f"""営業シナリオ作成AIです。
業界: {industry}、取引先: {client_data.get('name')}
ユーザー特性: {est_str}
フェーズ: {current['phase']} (評価軸: {current['axis']})

【直近の経緯】
{history_str}

直近の選択を踏まえた顧客の反応と、次のフェーズの2択行動を生成してください。
【文字数制限】状況の解像度を上げるため、situationは120〜150文字程度で顧客の反応や感情を具体的に描写し、option_a/bはそれぞれ60文字程度で記述すること。
【出力形式】必ずJSONのみ。
{{
  "phase": "{current['phase']}",
  "axis": "{current['axis']}",
  "situation": "【前の選択を受けた顧客の具体的な反応と新たな発言】",
  "question": "次にどう対応しますか？",
  "option_a": "【行動A】",
  "option_b": "【行動B】"
}}"""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        res = model.generate_content(prompt, generation_config=genai.GenerationConfig(response_mime_type="application/json"))
        return json.loads(res.text)
    except Exception as e:
        return {
            "client_name": client_data.get("name", "通信エラー株式会社") if step_index > 0 else "通信エラー株式会社",
            "client_context": client_data.get("context", "通信エラーが発生しました。") if step_index > 0 else "通信エラーが発生しました。",
            "phase": current["phase"],
            "axis": current["axis"],
            "situation": "ネットワークエラーで商談が中断されました。接続を確認してください。",
            "question": "どう対応しますか？",
            "option_a": "もう一度試す",
            "option_b": "少し待つ"
        }

# --- Gemini API (STEP4: 総合フィードバックの生成) ---
def generate_final_feedback(estimates: list[dict], aptitude_score: int, industry: str) -> dict:
    est_str = ", ".join([f"{e['axis']}={e['theta']} ({e['level']})" for e in estimates])
    prompt = f"""あなたは優秀なキャリアアドバイザーです。
就活生が適性診断と営業ロールプレイを終えました。以下のデータに基づき、総合的なフィードバックを作成してください。

志望業界: {industry}
営業適性スコア: {aptitude_score}%
各特性のIRT推定値: {est_str}

【条件】
- 200〜300文字程度で簡潔かつ温かみのあるトーンで記述すること。
- IRTのスコアに基づき、その人の「強み」と「今後意識すべきアドバイス」を必ず含めること。
- 必ず以下のJSONフォーマットのみで出力すること。
{{
    "feedback": "【フィードバックの本文】"
}}"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        res = model.generate_content(prompt, generation_config=genai.GenerationConfig(response_mime_type="application/json"))
        return json.loads(res.text)
    except Exception as e:
        return {"feedback": "フィードバックの生成に失敗しました。素晴らしいポテンシャルを持っていますので、引き続き頑張ってください！"}


# ==========================================
# フロントエンド (Figma Memphis UI/UX)
# ==========================================
app = Flask(__name__, static_folder='static')

BASE_CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: "Noto Sans JP", sans-serif;
  background-color: #fdf6ec;
  color: #1a1a1a;
  min-height: 100vh;
  overflow-x: hidden;
}

.deco { position: fixed; pointer-events: none; z-index: 0; }
.deco-circle-outline { width: 340px; height: 340px; border: 6px solid #ff3b2f; border-radius: 50%; top: -100px; right: -80px; opacity: 0.12; }
.deco-circle-fill { width: 64px; height: 64px; background: #ffe400; border-radius: 50%; top: 140px; right: 80px; opacity: 0.45; }
.deco-square-outline { width: 220px; height: 220px; border: 5px solid #0050ff; bottom: 20px; left: -70px; transform: rotate(20deg); opacity: 0.10; }
.deco-dots { bottom: 50px; right: 50px; display: grid; grid-template-columns: repeat(3, 9px); gap: 13px; opacity: 0.10; }
.deco-dots span { display: block; width: 9px; height: 9px; background: #1a1a1a; border-radius: 50%; }
.deco-zigzag { top: 80px; left: 0; opacity: 0.15; }

.container { position: relative; z-index: 1; max-width: 520px; margin: 0 auto; padding: 48px 16px 64px; }

.step-label {
  display: inline-block;
  font-family: "Dela Gothic One", sans-serif;
  font-size: 11px; letter-spacing: 0.2em; text-transform: uppercase;
  border: 2px solid #1a1a1a; padding: 4px 10px; margin-bottom: 12px;
  background: #fff;
}
.subtitle { font-size: 13px; color: #6b5e52; margin-bottom: 24px; font-weight: 700; }
.progress-count { font-family: "Dela Gothic One", sans-serif; font-size: 16px; color: #1a1a1a; margin-bottom: 24px; display: block; }

.card { background: #fff; border: 2px solid #1a1a1a; padding: 24px; margin-bottom: 20px; border-radius: 8px;}
.q-card { display: none; opacity: 0; transform: translateY(10px); transition: all 0.3s ease; }
.q-card.active { opacity: 1; transform: translateY(0); }

.card-meta { display: flex; align-items: baseline; gap: 12px; margin-bottom: 16px; justify-content: space-between; }
.card-num { font-family: "Dela Gothic One", sans-serif; font-size: 36px; line-height: 1; color: var(--card-color); }
.card-text { font-size: 15px; font-weight: 700; line-height: 1.7; margin-bottom: 24px; }

.btn-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.btn-answer { position: relative; display: block; }
.btn-answer input { position: absolute; opacity: 0; cursor: pointer; height: 0; width: 0; }
.btn-answer span {
  display: flex; justify-content: center; align-items: center; text-align: left;
  padding: 16px; font-family: "Noto Sans JP", sans-serif; font-size: 14px; font-weight: 700;
  background: #fff; border: 2px solid #1a1a1a; border-radius: 8px;
  color: #1a1a1a; cursor: pointer; transition: all 0.1s;
  box-shadow: 3px 3px 0px #1a1a1a; 
}
.btn-answer input:checked + span {
  background: var(--card-color); color: #fff;
  transform: translate(3px, 3px); box-shadow: 0px 0px 0px #1a1a1a; border-color: #1a1a1a;
}
.btn-answer span:hover { background: #fdf6ec; }
.btn-answer input:checked + span:hover { background: var(--card-color); }

.btn-submit {
  width: 100%; padding: 18px; font-family: "Dela Gothic One", sans-serif; font-size: 16px;
  background: #1a1a1a; border: 2px solid #1a1a1a; color: #fff; border-radius: 8px;
  cursor: pointer; transition: all 0.1s; margin-top: 16px;
  box-shadow: 4px 4px 0px rgba(0,0,0,0.2);
}
.btn-submit:hover { transform: translate(2px, 2px); box-shadow: 2px 2px 0px rgba(0,0,0,0.2); }
.btn-secondary { background: #fff; color: #1a1a1a; box-shadow: 4px 4px 0px #1a1a1a; }
.btn-secondary:hover { background: #fdf6ec; transform: translate(2px, 2px); box-shadow: 2px 2px 0px #1a1a1a; }

.chart-container { background: #fff; border: 2px solid #1a1a1a; border-radius: 8px; padding: 20px; margin-bottom: 24px; }
.result-badge {
  border: 2px solid #1a1a1a; padding: 32px; margin-bottom: 24px; border-radius: 8px;
  position: relative; overflow: hidden; text-align: center; background: #0050ff;
  box-shadow: 4px 4px 0px #1a1a1a;
}
.result-badge::before { content: ""; position: absolute; top: 12px; right: 12px; width: 40px; height: 40px; background: rgba(255,255,255,0.15); transform: rotate(12deg); }
.result-badge::after { content: ""; position: absolute; bottom: 12px; left: 12px; width: 56px; height: 56px; border: 4px solid rgba(255,255,255,0.15); transform: rotate(-8deg); }
.result-emoji { font-size: 64px; margin-bottom: 12px; font-family: "Dela Gothic One", sans-serif; color: #fff; line-height: 1; text-shadow: 2px 2px 0px #1a1a1a; }
.result-type { font-family: "Dela Gothic One", sans-serif; font-size: 14px; color: #fff; margin-bottom: 4px; }
"""

DECO_HTML = """
  <div class="deco deco-circle-outline"></div><div class="deco deco-circle-fill"></div><div class="deco deco-square-outline"></div>
  <div class="deco deco-dots"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div>
  <svg class="deco deco-zigzag" width="110" height="36" viewBox="0 0 110 36"><polyline points="0,18 14,4 28,18 42,4 56,18 70,4 84,18 98,4 110,18" fill="none" stroke="#9b00ff" stroke-width="3.5"/></svg>
"""

# ==========================================
# React ロード画面
# ==========================================
LOADING_HTML = """
<script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@700;800&display=swap" rel="stylesheet" />
<style>
  @keyframes bounce-dot {
    0%, 80%, 100% { transform: translateY(0); opacity: .4; }
    40% { transform: translateY(-10px); opacity: 1; }
  }
  @keyframes sweep {
    0% { transform: translateX(-110%); }
    100% { transform: translateX(380%); }
  }
  #loading-overlay {
    display: none;
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    z-index: 9999;
    background: linear-gradient(160deg,#FFFBEB 0%,#FEF3C7 55%,#FDE68A 100%);
  }
  body.loading-active {
    overflow: hidden;
  }
</style>

<div id="loading-overlay">
  <div id="loading-root" style="width: 100%; height: 100%;"></div>
</div>

<script>
  function showLoading() {
    document.body.classList.add('loading-active');
    document.getElementById('loading-overlay').style.display = 'block';
  }
</script>

<script type="text/babel">
  const { useEffect, useRef, useState } = React;

  const PATH_D = [
    "M 20,10 L 20,70 Q 20,82 32,82 L 80,82",
    "M 135,10 A 41,37 0 0 1 176,47 A 41,37 0 0 1 135,84 A 41,37 0 0 1 94,47 A 41,37 0 0 1 135,10",
    "M 195,82 L 230,10 L 265,82 L 250,57 L 210,57",
    "M 280,10 L 280,82 Q 280,82 322,82 Q 364,82 364,47 Q 364,12 322,10 Q 280,10 280,10",
    "M 385,10 L 385,82",
    "M 412,82 L 412,10 L 490,82 L 490,10",
    "M 583,22 Q 565,8 540,14 Q 514,22 514,47 Q 514,72 540,78 Q 565,86 583,72 L 583,47 L 548,47",
  ].join(" ");

  function Baby({ phase }) {
    const p = phase === 0;
    const ink = "#2A1F14", sw = 1.6;
    const skin = "#EEE4CC", blue = "#5B90C8", bd = "#4878AA";
    return (
      <g>
        <ellipse cx={p?-8:-5} cy={5} rx={5.5} ry={4} fill={blue} stroke={ink} strokeWidth={sw} transform={`rotate(${p?-25:5})`} />
        <ellipse cx={2} cy={4} rx={13} ry={10} fill={blue} stroke={ink} strokeWidth={sw} />
        <ellipse cx={p?10:7} cy={13} rx={7} ry={4.5} fill={blue} stroke={ink} strokeWidth={sw} transform={`rotate(${p?10:-20},${p?10:7},13)`} />
        <ellipse cx={p?5:9} cy={14} rx={7} ry={4.5} fill={bd} stroke={ink} strokeWidth={sw} transform={`rotate(${p?-20:10},${p?5:9},14)`} />
        <circle cx={18} cy={-6} r={16} fill={skin} stroke={ink} strokeWidth={sw} />
        <ellipse cx={5} cy={-5} rx={3} ry={4} fill={skin} stroke={ink} strokeWidth={sw} />
        <circle cx={12} cy={-10} r={2.4} fill={ink} />
        <circle cx={22} cy={-9} r={2.4} fill={ink} />
        <circle cx={13} cy={-11.2} r={0.8} fill="white" />
        <circle cx={23} cy={-10.2} r={0.8} fill="white" />
        <circle cx={10} cy={-4} r={4.5} fill="#F0905A" opacity={0.6} />
        <circle cx={26} cy={-3} r={4.5} fill="#F0905A" opacity={0.6} />
        <ellipse cx={8} cy={-6} rx={2} ry={1.4} fill="#D4956A" />
        <path d="M9,-1 Q16,4 23,-1" stroke={ink} strokeWidth={2} fill="none" strokeLinecap="round" />
        <ellipse cx={p?-4:-7} cy={6} rx={5.5} ry={4} fill={bd} stroke={ink} strokeWidth={sw} transform={`rotate(${p?5:-25})`} />
      </g>
    );
  }

  function LoadingApp() {
    const pathRef = useRef(null);
    const rafRef = useRef(0);
    const progRef = useRef(0);
    const tickRef = useRef(0);
    const [baby, setBaby] = useState({ x: 20, y: 10, goingLeft: false, phase: 0 });
    const [dots, setDots] = useState(0);

    useEffect(() => {
      const run = () => {
        const path = pathRef.current;
        if (path) {
          const total = path.getTotalLength();
          progRef.current = (progRef.current + 1.6) % total;
          tickRef.current++;
          const p = progRef.current, p2 = (p + 3) % total;
          const pt = path.getPointAtLength(p), pt2 = path.getPointAtLength(p2);
          const dx = pt2.x - pt.x, dy = pt2.y - pt.y;
          const phase = Math.floor(tickRef.current / 7) % 2;
          if (Math.hypot(dx, dy) < 20)
            setBaby({ x: pt.x, y: pt.y, goingLeft: dx < -0.5, phase });
        }
        rafRef.current = requestAnimationFrame(run);
      };
      rafRef.current = requestAnimationFrame(run);
      return () => cancelAnimationFrame(rafRef.current);
    }, []);

    useEffect(() => {
      const t = setInterval(() => setDots(d => (d + 1) % 4), 500);
      return () => clearInterval(t);
    }, []);

    return (
      <div style={{ width:"100%", height:"100%", display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:36 }}>
        <div style={{ width:"min(96vw,660px)" }}>
          <svg viewBox="0 0 620 95" style={{ width:"100%", height:"auto", overflow:"visible" }}>
            <defs>
              <filter id="sh" x="-10%" y="-20%" width="130%" height="160%">
                <feDropShadow dx="2" dy="4" stdDeviation="3" floodColor="#D97706" floodOpacity="0.35" />
              </filter>
            </defs>
            <path d={PATH_D} fill="none" stroke="#FCD34D" strokeWidth={12} strokeLinecap="round" strokeLinejoin="round" filter="url(#sh)" />
            <path d={PATH_D} fill="none" stroke="white" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" opacity={0.28} />
            <path ref={pathRef} d={PATH_D} fill="none" stroke="none" />
            <g transform={`translate(${baby.x},${baby.y}) scale(${baby.goingLeft?-1:1},1)`} style={{ filter:"drop-shadow(0 2px 5px rgba(0,0,0,0.2))" }}>
              <Baby phase={baby.phase} />
            </g>
          </svg>
        </div>
        <p style={{ fontFamily:"'Nunito',sans-serif", fontWeight:700, fontSize:"clamp(14px,3vw,18px)", color:"#92400E", opacity:0.75, letterSpacing:"0.05em" }}>
          {"AIが考えています" + "・".repeat(dots)}
        </p>
        <div style={{ width:"min(380px,78%)", height:6, borderRadius:99, background:"#FDE68A", overflow:"hidden" }}>
          <div style={{ height:"100%", width:"35%", borderRadius:99, background:"linear-gradient(90deg,#FBBF24,#F59E0B)", animation:"sweep 2s ease-in-out infinite" }} />
        </div>
        <div style={{ display:"flex", gap:12 }}>
          {[0,1,2].map(i => (
            <div key={i} style={{ width:10, height:10, borderRadius:"50%", background:"#FBBF24", animation:`bounce-dot 1.2s ease-in-out ${i*0.22}s infinite` }} />
          ))}
        </div>
      </div>
    );
  }

  ReactDOM.createRoot(document.getElementById("loading-root")).render(<LoadingApp />);
</script>
"""

START_TEMPLATE = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>営業適性診断 & AIロープレ</title>
  <link href="https://fonts.googleapis.com/css2?family=Dela+Gothic+One&family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">
  <style>{{ css }}</style>
</head>
<body>
  {{ deco|safe }}
  <main class="container" style="text-align: center; padding-top: 15vh;">
    <div class="step-label" style="border-color:#ff3b2f; color:#ff3b2f;">STEP 0</div>
    <h1 style="font-family:'Dela Gothic One', sans-serif; font-size:36px; color:#1a1a1a; margin-bottom:24px; line-height:1.2;">営業適性診断<br>& AIロープレ</h1>
    <p style="font-size:15px; color:#6b5e52; font-weight:700; margin-bottom:56px; line-height:1.8;">
      基本診断とAIによる独自深掘り質問、さらに実践的なロールプレイで、<br>
      あなたの営業としての強みと思考性を分析します。
    </p>
    <a href="{{ url_for('step1') }}" style="text-decoration:none;">
      <button class="btn-submit" style="background:#ff3b2f; border-color:#ff3b2f; font-size: 20px; padding: 24px;">診断をスタート →</button>
    </a>
  </main>
</body></html>
"""

INDEX_TEMPLATE = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>STEP 1: 営業適性 基本診断</title>
  <link href="https://fonts.googleapis.com/css2?family=Dela+Gothic+One&family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">
  <style>{{ css }}</style>
</head>
<body>
  {{ deco|safe }}
  {{ loading_html|safe }}
  <main class="container">
    <div class="step-label">STEP 1: 基本診断</div>
    <p class="subtitle">直感で「はい」か「いいえ」でお答えください。</p>
    <span class="progress-count" id="progress-count">PAGE 1 / 5</span>

    <form method="post" action="{{ url_for('step1_custom') }}" id="quiz-form" onsubmit="showLoading();">
      {% set colors = ['#ff3b2f', '#0050ff', '#00c26a', '#ff7d00', '#9b00ff'] %}
      {% for q in questions %}
        {% set c = colors[loop.index0 % 5] %}
        <div class="card q-card" data-index="{{ loop.index0 }}" style="--card-color: {{ c }};">
          <div class="card-meta">
            <span class="card-num">{{ "%02d"|format(loop.index) }}</span>
          </div>
          <p class="card-text">{{ q.text }}</p>
          <div class="btn-row" style="grid-template-columns: 1fr 1fr; gap:12px;">
            <label class="btn-answer"><input required type="radio" name="{{ q.id }}" value="yes"><span style="justify-content:center;">はい</span></label>
            <label class="btn-answer"><input required type="radio" name="{{ q.id }}" value="no"><span style="justify-content:center;">いいえ</span></label>
          </div>
        </div>
      {% endfor %}
      
      <div id="nav-container" style="margin-top: 24px;">
        <button type="button" class="btn-submit" id="next-btn">次の4問へ →</button>
        <button type="submit" class="btn-submit" id="submit-btn" style="display:none; background:#ff3b2f; border-color:#ff3b2f;">深掘り独自質問へ進む →</button>
      </div>
    </form>
  </main>
  
  <script>
    document.addEventListener("DOMContentLoaded", () => {
      const cards = Array.from(document.querySelectorAll('.q-card'));
      const nextBtn = document.getElementById('next-btn');
      const submitBtn = document.getElementById('submit-btn');
      const progressCount = document.getElementById('progress-count');
      
      const itemsPerPage = 4;
      const totalPages = Math.ceil(cards.length / itemsPerPage);
      let currentPage = 0;

      function updateUI() {
        cards.forEach((card, i) => {
          if (i >= currentPage * itemsPerPage && i < (currentPage + 1) * itemsPerPage) {
            card.style.display = 'block';
            setTimeout(() => card.classList.add('active'), 10);
          } else {
            card.style.display = 'none';
            card.classList.remove('active');
          }
        });
        
        progressCount.innerText = `PAGE ${currentPage + 1} / ${totalPages}`;
        
        if (currentPage === totalPages - 1) {
          nextBtn.style.display = 'none';
          submitBtn.style.display = 'block';
        } else {
          nextBtn.style.display = 'block';
          submitBtn.style.display = 'none';
        }
      }

      nextBtn.addEventListener('click', () => {
        const currentCards = cards.slice(currentPage * itemsPerPage, (currentPage + 1) * itemsPerPage);
        const allAnswered = currentCards.every(card => card.querySelector('input[type="radio"]:checked'));
        
        if (!allAnswered) {
          alert('表示されているすべての質問に回答してください！');
          return;
        }
        currentPage++;
        updateUI();
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });

      updateUI();
    });
  </script>
</body></html>
"""

CUSTOM_QS_TEMPLATE = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>STEP 1.5: 深掘り独自質問</title>
  <link href="https://fonts.googleapis.com/css2?family=Dela+Gothic+One&family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">
  <style>{{ css }}</style>
</head>
<body>
  {{ deco|safe }}
  {{ loading_html|safe }}
  <main class="container">
    <div class="step-label" style="border-color:#ff7d00; color:#ff7d00;">STEP 1.5: 深掘り診断</div>
    <p class="subtitle">AIがあなたの回答傾向から、精度を高めるための5問を生成しました。</p>

    <form method="post" action="{{ url_for('result') }}" id="custom-quiz-form" onsubmit="showLoading();">
      {% for key, val in base_answers.items() %}
      <input type="hidden" name="{{ key }}" value="{{ val }}">
      {% endfor %}
      <input type="hidden" name="custom_qs_json" value="{{ custom_qs_json }}">
      
      {% set colors = ['#ff7d00', '#ff3b2f', '#0050ff', '#00c26a', '#9b00ff'] %}
      {% for q in custom_qs %}
        {% set c = colors[loop.index0 % 5] %}
        <div class="card" style="--card-color: {{ c }};">
          <div class="card-meta">
            <span class="card-num">C{{ loop.index }}</span>
          </div>
          <p class="card-text">{{ q.text }}</p>
          <div class="btn-row" style="grid-template-columns: 1fr 1fr; gap:12px;">
            <label class="btn-answer"><input required type="radio" name="{{ q.id }}" value="yes"><span style="justify-content:center;">はい</span></label>
            <label class="btn-answer"><input required type="radio" name="{{ q.id }}" value="no"><span style="justify-content:center;">いいえ</span></label>
          </div>
        </div>
      {% endfor %}
      
      <div style="margin-top: 24px;">
        <button type="submit" class="btn-submit" style="background:#ff7d00; border-color:#ff7d00;">診断結果を見る →</button>
      </div>
    </form>
  </main>
</body></html>
"""

RESULT_TEMPLATE = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>STEP 2: 分析結果</title>
  <link href="https://fonts.googleapis.com/css2?family=Dela+Gothic+One&family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    {{ css }}
    .image-selector {
      border: 2px solid #1a1a1a;
      width: 64px; height: 64px;
      object-fit: cover;
      border-radius: 8px;
      transition: 0.1s;
    }
    input[type="radio"]:checked + .image-selector {
      border: 3px solid #9b00ff !important;
      transform: scale(1.1);
    }
  </style>
</head>
<body>
  {{ deco|safe }}
  {{ loading_html|safe }}
  <main class="container">
    <div class="step-label">RESULT</div>
    
    <div class="result-badge" {% if aptitude_score >= 80 %}style="background: #ff3b2f;"{% endif %}>
      <div class="result-type">SALES APTITUDE MATCH</div>
      <div class="result-emoji">{{ aptitude_score }}%</div>
    </div>

    <div class="chart-container">
      <canvas id="radarChart"></canvas>
    </div>

    <form method="post" action="{{ url_for('ai_start') }}" id="ai-form" onsubmit="showLoading();">
      <input type="hidden" name="estimates_data" value="{{ estimates_json }}">
      
      <div class="card" style="padding: 16px; border-color: #9b00ff; margin-bottom: 16px;">
        <label style="font-family:'Dela Gothic One', sans-serif; font-size:12px; color:#9b00ff; display:block; margin-bottom:8px;">★ 志望業界を選択して実践ロープレへ</label>
        <select name="industry" required style="width:100%; padding:12px; border:2px solid #1a1a1a; border-radius:8px; font-family:'Noto Sans JP', sans-serif; font-weight:700; font-size:14px; cursor:pointer; margin-bottom: 24px;">
          <option value="IT/SaaS">IT / SaaS</option>
          <option value="人材">人材サービス</option>
          <option value="保険">金融・保険</option>
          <option value="メーカー">メーカー（製造業）</option>
          <option value="広告">広告・代理店</option>
          <option value="不動産">不動産</option>
        </select>

        <label style="font-family:'Dela Gothic One', sans-serif; font-size:12px; color:#9b00ff; display:block; margin-bottom:8px;">★ 商談相手の人物を選択</label>
        <div style="display: flex; gap: 12px; overflow-x: auto; padding-bottom: 8px; justify-content: space-between;">
          {% for img in client_images %}
          <label style="cursor: pointer; flex-shrink: 0;">
            <input type="radio" name="client_image" value="{{ img }}" {% if loop.first %}checked{% endif %} style="display: none;">
            <img src="{{ url_for('static', filename=img) }}" class="image-selector">
          </label>
          {% endfor %}
        </div>
      </div>

      <button type="submit" class="btn-submit" id="ai-btn" style="background:#9b00ff; border-color:#9b00ff;">🤖 AIロープレに進む</button>
      <a href="{{ url_for('start') }}" style="display:block; text-decoration:none;"><button type="button" class="btn-submit btn-secondary">← 最初からやり直す</button></a>
    </form>
  </main>
  
  <script>
    const ctx = document.getElementById('radarChart').getContext('2d');
    new Chart(ctx, {
      type: 'radar',
      data: {
        labels: [{% for e in estimates %}"{{ e.axis }}",{% endfor %}],
        datasets: [
          {
            label: 'あなた',
            data: [{% for e in estimates %}{{ e.theta }},{% endfor %}],
            backgroundColor: 'rgba(0, 80, 255, 0.2)',
            borderColor: '#0050ff',
            borderWidth: 3,
            pointBackgroundColor: '#0050ff',
            pointRadius: 4
          },
          {
            label: '優秀な営業（理想）',
            data: [{% for e in estimates %}{{ ideal_profile[e.axis] }},{% endfor %}],
            backgroundColor: 'transparent',
            borderColor: '#ff3b2f',
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 0
          }
        ]
      },
      options: {
        scales: {
          r: {
            min: -3, max: 3,
            ticks: { display: false },
            grid: { color: 'rgba(26,26,26,0.1)' },
            angleLines: { color: 'rgba(26,26,26,0.1)' },
            pointLabels: { font: { family: "'Noto Sans JP', sans-serif", weight: 'bold', size: 11 }, color: '#1a1a1a' }
          }
        },
        plugins: {
          legend: { position: 'bottom', labels: { font: { family: "'Noto Sans JP', sans-serif", weight: 'bold' } } }
        }
      }
    });
  </script>
</body></html>
"""

AI_STEP_TEMPLATE = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>STEP 3: AIロールプレイ ({{ step_index + 1 }}/5)</title>
  <link href="https://fonts.googleapis.com/css2?family=Dela+Gothic+One&family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">
  <style>{{ css }}</style>
</head>
<body>
  {{ deco|safe }}
  {{ loading_html|safe }}
  <main class="container">
    <div class="step-label" style="border-color:#9b00ff; color:#9b00ff;">STEP 3: AI ROLEPLAY ({{ industry }})</div>
    <span class="progress-count" style="color:#9b00ff;">PHASE {{ step_index + 1 }} / 5</span>

    <div class="card" style="border-color: #0050ff; background: #f0f5ff; margin-bottom: 24px;">
      <p style="font-size: 13px; color: #0050ff; font-weight: 700; margin-bottom: 8px;">🏢 商談相手</p>
      <img src="{{ url_for('static', filename=client_image) }}" style="width: 100%; aspect-ratio: 1/1; object-fit: cover; border-radius: 8px; border: 2px solid #1A1A1A; margin-bottom: 12px;">
      <h2 style="font-size: 18px; margin-bottom: 8px; font-weight: 700;">{{ client_name }}</h2>
      <p style="font-size: 14px; line-height: 1.6; color: #1a1a1a;">{{ client_context }}</p>
    </div>

    <form method="post" action="{{ url_for('ai_process') }}" id="ai-quiz-form" onsubmit="showLoading();">
      <input type="hidden" name="estimates_data" value="{{ estimates_json }}">
      <input type="hidden" name="industry" value="{{ industry }}">
      <input type="hidden" name="client_data" value="{{ client_data_json }}">
      <input type="hidden" name="history" value="{{ history_json }}">
      <input type="hidden" name="step_index" value="{{ step_index }}">

      <input type="hidden" name="option_a_text" value="{{ q.option_a }}">
      <input type="hidden" name="option_b_text" value="{{ q.option_b }}">
      <input type="hidden" name="situation" value="{{ q.situation }}">
      <input type="hidden" name="phase_text" value="{{ q.phase }}">
      
      <div class="card" style="--card-color: #9b00ff;">
        <div class="card-meta">
          <span class="card-num">Q{{ step_index + 1 }}</span>
          <span style="font-size:12px; font-weight:700; color:#9b00ff; border: 2px solid #9b00ff; padding: 4px 8px; border-radius: 4px;">{{ q.phase }}</span>
        </div>
        
        <div style="background:#fdf6ec; padding:16px; border-radius:8px; border: 2px dashed #1a1a1a; margin-bottom:16px;">
          <p style="font-size:12px; color:#6b5e52; margin-bottom:8px; font-weight:700;">【顧客の状況・発言】</p>
          <p style="font-size:15px; font-weight:700; line-height:1.8; color:#1a1a1a;">「{{ q.situation }}」</p>
        </div>

        <p class="card-text" style="text-align:center; margin-bottom:16px;">{{ q.question }}</p>
        
        <div class="btn-row" style="grid-template-columns: 1fr; gap:16px;">
          <label class="btn-answer">
            <input required type="radio" name="user_choice" value="A">
            <span style="text-align:left; justify-content:flex-start; line-height:1.5;">A. {{ q.option_a }}</span>
          </label>
          <label class="btn-answer">
            <input required type="radio" name="user_choice" value="B">
            <span style="text-align:left; justify-content:flex-start; line-height:1.5;">B. {{ q.option_b }}</span>
          </label>
        </div>
      </div>
      
      <div id="submit-container" style="display:block;">
        <button type="submit" class="btn-submit" id="next-submit-btn" style="background:#9b00ff; border-color:#9b00ff;">
          {% if step_index < 4 %}次の展開へ進む →{% else %}最終結果を見る →{% endif %}
        </button>
      </div>
    </form>
  </main>
</body></html>
"""

FEEDBACK_TEMPLATE = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>STEP 4: 総合フィードバック</title>
  <link href="https://fonts.googleapis.com/css2?family=Dela+Gothic+One&family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">
  <style>{{ css }}</style>
</head>
<body>
  {{ deco|safe }}
  <main class="container">
    <div class="step-label" style="border-color:#00c26a; color:#00c26a;">STEP 4: FINAL FEEDBACK</div>
    <p class="subtitle">AIによるあなたの適性分析とアドバイス</p>

    <div class="card" style="border: 4px solid #00c26a; padding: 32px 24px; box-shadow: 6px 6px 0px #00c26a; margin-bottom: 32px;">
      <h2 style="font-size: 18px; margin-bottom: 16px; font-weight: 700; color:#00c26a;">🎯 総合アドバイス</h2>
      <div style="font-size:15px; line-height:2.0; font-weight:700; white-space:pre-wrap; color:#1a1a1a;">{{ final_feedback }}</div>
    </div>

    <a href="{{ url_for('start') }}" style="text-decoration:none;"><button class="btn-submit btn-secondary">← 最初からやり直す</button></a>
  </main>
</body></html>
"""

# --- ルーティング ---
@app.get("/")
def start():
    return render_template_string(START_TEMPLATE, css=BASE_CSS, deco=DECO_HTML)

@app.get("/step1")
def step1():
    return render_template_string(INDEX_TEMPLATE, css=BASE_CSS, deco=DECO_HTML, questions=QUESTIONS, loading_html=LOADING_HTML)

@app.post("/step1_custom")
def step1_custom():
    base_answers = {q["id"]: request.form.get(q["id"]) for q in QUESTIONS}
    if any(a not in {"yes", "no"} for a in base_answers.values()): return redirect(url_for("step1"))
    
    initial_estimates = estimate_thetas(base_answers, QUESTIONS)
    custom_qs = generate_custom_questions(initial_estimates)
    
    return render_template_string(
        CUSTOM_QS_TEMPLATE, 
        css=BASE_CSS, 
        deco=DECO_HTML, 
        base_answers=base_answers,
        custom_qs=custom_qs,
        custom_qs_json=json.dumps(custom_qs),
        loading_html=LOADING_HTML
    )

@app.post("/result")
def result():
    base_answers = {q["id"]: request.form.get(q["id"]) for q in QUESTIONS}
    custom_qs_json = request.form.get("custom_qs_json")
    custom_qs = json.loads(custom_qs_json) if custom_qs_json else []
    custom_answers = {q["id"]: request.form.get(q["id"]) for q in custom_qs}
    
    all_answers = {**base_answers, **custom_answers}
    all_questions = QUESTIONS + custom_qs
    
    estimates = estimate_thetas(all_answers, all_questions)
    aptitude_score = calculate_sales_aptitude(estimates)
    
    return render_template_string(
        RESULT_TEMPLATE, 
        css=BASE_CSS, 
        deco=DECO_HTML, 
        estimates=estimates, 
        estimates_json=json.dumps(estimates), 
        aptitude_score=aptitude_score, 
        ideal_profile=IDEAL_PROFILE,
        loading_html=LOADING_HTML,
        client_images=CLIENT_IMAGES
    )

@app.post("/ai_start")
def ai_start():
    est_json = request.form.get("estimates_data")
    industry = request.form.get("industry", "IT/SaaS")
    client_image = request.form.get("client_image", CLIENT_IMAGES[0])
    if not est_json: return redirect(url_for("start"))
    
    ai_data = generate_roleplay_step(industry, json.loads(est_json), 0, {}, [])
    
    client_name = ai_data.get("client_name", "テスト企業")
    client_context = ai_data.get("client_context", "背景情報の読み込みに失敗しました。")
    client_data_json = json.dumps({"name": client_name, "context": client_context, "image": client_image}, ensure_ascii=False)
    
    return render_template_string(
        AI_STEP_TEMPLATE, 
        css=BASE_CSS, 
        deco=DECO_HTML, 
        q=ai_data, 
        estimates_json=est_json, 
        industry=industry,
        client_name=client_name,
        client_context=client_context,
        client_image=client_image,
        client_data_json=client_data_json,
        history_json=json.dumps([]),
        step_index=0,
        loading_html=LOADING_HTML
    )

@app.post("/ai_process")
def ai_process():
    est_json = request.form.get("estimates_data")
    industry = request.form.get("industry")
    client_data_json = request.form.get("client_data")
    history_json = request.form.get("history")
    step_index = int(request.form.get("step_index", 0))

    user_choice = request.form.get("user_choice")
    option_text_a = request.form.get("option_a_text")
    option_text_b = request.form.get("option_b_text")
    situation = request.form.get("situation")
    phase_text = request.form.get("phase_text")

    if not est_json: return redirect(url_for("start"))

    chosen_text = option_text_a if user_choice == "A" else option_text_b
    history = json.loads(history_json)
    history.append({
        "phase": phase_text,
        "situation": situation,
        "chosen_option": f"{user_choice}: {chosen_text}"
    })

    step_index += 1

    # 5問完了したらSTEP4:総合フィードバックを生成して遷移
    if step_index >= 5:
        estimates = json.loads(est_json)
        aptitude_score = calculate_sales_aptitude(estimates)
        
        # AIによるフィードバック生成
        ai_fb_data = generate_final_feedback(estimates, aptitude_score, industry)
        final_feedback = ai_fb_data.get("feedback", "フィードバックの生成に失敗しました。")
        
        return render_template_string(FEEDBACK_TEMPLATE, css=BASE_CSS, deco=DECO_HTML, final_feedback=final_feedback)

    client_data = json.loads(client_data_json)
    client_image = client_data.get("image", CLIENT_IMAGES[0])
    ai_data = generate_roleplay_step(industry, json.loads(est_json), step_index, client_data, history)

    return render_template_string(
        AI_STEP_TEMPLATE, 
        css=BASE_CSS, 
        deco=DECO_HTML, 
        q=ai_data, 
        estimates_json=est_json, 
        industry=industry,
        client_name=client_data["name"],
        client_context=client_data["context"],
        client_image=client_image,
        client_data_json=client_data_json,
        history_json=json.dumps(history, ensure_ascii=False),
        step_index=step_index,
        loading_html=LOADING_HTML
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
    